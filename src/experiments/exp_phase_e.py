# --------------------------------------------------
# src/experiments/exp_phase_e.py
# --------------------------------------------------
from __future__ import annotations

from src.core.context import (
    initialize_context_state,
    update_context_state,
)
from src.core.field_density import (
    build_field_axis,
    compute_density_variance,
    find_density_peaks,
    reconstruct_density,
)
from src.core.mcm_state import (
    compute_state_velocity,
    initialize_mcm_state,
    update_mcm_state,
)
from src.core.neural_core import (
    build_neural_core,
    decode_neural_state,
    step_neural_core,
)
from src.core.output import build_output_state
from src.core.perception import apply_input_gain, encode_perception
from src.core.reflection import build_reflection_state
from src.core.self_state import build_self_state, build_self_state_labels


# --------------------------------------------------
# Phase-E-Inputs
# --------------------------------------------------
def build_phase_e_inputs() -> list[dict[str, float]]:
    return [
        {
            "valence": 0.9,
            "novelty": 0.5,
            "relevance": 0.8,
            "uncertainty": 0.1,
            "social_salience": 0.0,
        },
        {
            "valence": 0.2,
            "novelty": 0.1,
            "relevance": 0.3,
            "uncertainty": 0.0,
            "social_salience": 0.0,
        },
        {
            "valence": -0.9,
            "novelty": 0.6,
            "relevance": 0.8,
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
    ]


# --------------------------------------------------
# Cluster-ID-Helfer
# --------------------------------------------------
def _derive_cluster_id(raw_input: dict[str, float]) -> float:
    valence = float(raw_input["valence"])
    if valence > 0.1:
        return 1.0
    if valence < -0.1:
        return 2.0
    return 0.0


# --------------------------------------------------
# Phase-E-Experiment
# --------------------------------------------------
def run_phase_e_experiment() -> dict[str, object]:
    dt = 0.1
    regulation_gain = 0.4
    replay_input = 0.0
    recurrent_feedback = 0.0
    noise = 0.0
    input_gain = 1.0

    field_axis = build_field_axis()
    preferred_positions = (-2.0, -1.0, 0.0, 1.0, 2.0)

    neural_state = build_neural_core()
    x_prev = initialize_mcm_state()
    context_state = initialize_context_state()
    previous_self_state = {
        "center_distance": abs(x_prev),
    }
    step_outputs: list[dict[str, object]] = []

    for step_index, raw_input in enumerate(build_phase_e_inputs()):
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
        density_variance = compute_density_variance(
            density=density,
            field_axis=field_axis,
        )
        density_peaks = find_density_peaks(
            density=density,
            field_axis=field_axis,
        )

        self_state = build_self_state(
            x_value=x_curr,
            velocity=velocity,
            density_variance=density_variance,
            cluster_stability=1.0 / (1.0 + float(len(density_peaks))),
        )
        self_labels = build_self_state_labels(self_state)

        context_state = update_context_state(
            previous_context=context_state,
            x_value=x_curr,
            density_variance=density_variance,
            cluster_id=_derive_cluster_id(raw_input),
            input_energy=abs(neural_drive),
            self_state=self_state,
        )
        reflection_state = build_reflection_state(
            x_value=x_curr,
            velocity=velocity,
            context_state=context_state,
            previous_self_state=previous_self_state,
            matched_cluster_ids=(_derive_cluster_id(raw_input),),
        )

        output_state = build_output_state(
            neural_state=neural_state,
            mcm_state=x_curr,
        )
        output_state["step"] = step_index
        output_state["velocity"] = velocity
        output_state["raw_input"] = dict(raw_input)
        output_state["self_state"] = dict(self_state)
        output_state["self_labels"] = list(self_labels)
        output_state["reflection_state"] = dict(reflection_state)

        step_outputs.append(output_state)
        previous_self_state = dict(self_state)
        x_prev = x_curr

    return {
        "dt": dt,
        "regulation_gain": regulation_gain,
        "field_axis": field_axis,
        "preferred_positions": preferred_positions,
        "steps": step_outputs,
    }


# --------------------------------------------------
# Phase-E-Auswertung
# --------------------------------------------------
def evaluate_phase_e_results(run_data: dict[str, object]) -> dict[str, object]:
    step_outputs = list(run_data["steps"])
    self_state_available = any(bool(step_output.get("self_state")) for step_output in step_outputs)
    self_labels_available = any(bool(step_output.get("self_labels")) for step_output in step_outputs)

    return {
        "self_state_available": self_state_available,
        "self_labels_available": self_labels_available,
        "num_steps": len(step_outputs),
    }