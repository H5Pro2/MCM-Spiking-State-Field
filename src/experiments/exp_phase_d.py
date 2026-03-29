# --------------------------------------------------
# src/experiments/exp_phase_d.py
# --------------------------------------------------
from __future__ import annotations

from src.core.context import (
    initialize_context_state,
    learn_context_transition,
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


# --------------------------------------------------
# Phase-D-Inputs
# --------------------------------------------------
def build_phase_d_inputs() -> list[dict[str, float]]:
    return [
        {
            "valence": 0.8,
            "novelty": 0.4,
            "relevance": 0.7,
            "uncertainty": 0.1,
            "social_salience": 0.0,
        },
        {
            "valence": 0.0,
            "novelty": 0.1,
            "relevance": 0.2,
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
            "valence": 0.0,
            "novelty": 0.0,
            "relevance": 0.0,
            "uncertainty": 0.0,
            "social_salience": 0.0,
        },
        {
            "valence": 0.8,
            "novelty": 0.3,
            "relevance": 0.6,
            "uncertainty": 0.1,
            "social_salience": 0.0,
        },
        {
            "valence": -0.8,
            "novelty": 0.4,
            "relevance": 0.6,
            "uncertainty": 0.2,
            "social_salience": 0.1,
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
# Phase-D-Experiment
# --------------------------------------------------
def run_phase_d_experiment() -> dict[str, object]:
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
    transition_memory: dict[str, float] = {}
    step_outputs: list[dict[str, object]] = []

    for step_index, raw_input in enumerate(build_phase_d_inputs()):
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

        previous_context = dict(context_state)
        context_state = update_context_state(
            previous_context=previous_context,
            x_value=x_curr,
            density_variance=density_variance,
            cluster_id=_derive_cluster_id(raw_input),
            input_energy=abs(neural_drive),
            self_state={"cluster_stability": 1.0 / (1.0 + float(len(density_peaks)))},
        )
        transition_memory = learn_context_transition(
            previous_context=previous_context,
            current_context=context_state,
            transition_memory=transition_memory,
        )

        output_state = build_output_state(
            neural_state=neural_state,
            mcm_state=x_curr,
        )
        output_state["step"] = step_index
        output_state["velocity"] = velocity
        output_state["raw_input"] = dict(raw_input)
        output_state["context_state"] = dict(context_state)
        output_state["transition_memory_size"] = len(transition_memory)

        step_outputs.append(output_state)
        x_prev = x_curr

    return {
        "dt": dt,
        "regulation_gain": regulation_gain,
        "field_axis": field_axis,
        "preferred_positions": preferred_positions,
        "steps": step_outputs,
        "context_state": context_state,
        "transition_memory": transition_memory,
    }


# --------------------------------------------------
# Phase-D-Auswertung
# --------------------------------------------------
def evaluate_phase_d_results(run_data: dict[str, object]) -> dict[str, object]:
    step_outputs = list(run_data["steps"])
    transition_memory = dict(run_data["transition_memory"])
    transition_count = int(sum(float(value) for value in transition_memory.values()))

    return {
        "context_built": len(step_outputs) > 0,
        "context_learning_active": transition_count > 0,
        "transition_count": transition_count,
        "num_steps": len(step_outputs),
    }