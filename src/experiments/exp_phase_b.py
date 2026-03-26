# --------------------------------------------------
# src/experiments/exp_phase_b.py
# --------------------------------------------------
from __future__ import annotations

from core.field_density import (
    build_field_axis,
    compute_density_mean,
    compute_density_variance,
    find_density_peaks,
    reconstruct_density,
)
from core.mcm_state import (
    compute_state_velocity,
    initialize_mcm_state,
    update_mcm_state,
)
from core.neural_core import (
    build_neural_core,
    decode_neural_state,
    step_neural_core,
)
from core.output import build_output_state
from core.perception import apply_input_gain, encode_perception


# --------------------------------------------------
# Phase-B-Inputs
# --------------------------------------------------
def build_phase_b_inputs() -> list[dict[str, float]]:
    return [
        {
            "valence": 0.0,
            "novelty": 0.0,
            "relevance": 0.0,
            "uncertainty": 0.0,
            "social_salience": 0.0,
        },
        {
            "valence": 0.9,
            "novelty": 0.6,
            "relevance": 0.8,
            "uncertainty": 0.2,
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
            "valence": 0.4,
            "novelty": 0.2,
            "relevance": 0.3,
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
            "valence": 0.0,
            "novelty": 0.0,
            "relevance": 0.0,
            "uncertainty": 0.0,
            "social_salience": 0.0,
        },
    ]


# --------------------------------------------------
# Phase-B-Experiment
# --------------------------------------------------
def run_phase_b_experiment() -> dict[str, object]:
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
    step_outputs: list[dict[str, object]] = []

    for step_index, raw_input in enumerate(build_phase_b_inputs()):
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

        step_outputs.append(output_state)
        x_prev = x_curr

    return {
        "dt": dt,
        "regulation_gain": regulation_gain,
        "field_axis": field_axis,
        "preferred_positions": preferred_positions,
        "steps": step_outputs,
    }


# --------------------------------------------------
# Phase-B-Auswertung
# --------------------------------------------------
def evaluate_phase_b_results(run_data: dict[str, object]) -> dict[str, object]:
    step_outputs = list(run_data["steps"])

    density_variances = [
        float(step_output["density_variance"])
        for step_output in step_outputs
    ]
    peak_counts = [
        len(step_output["density_peaks"])
        for step_output in step_outputs
    ]
    density_means = [
        float(step_output["density_mean"])
        for step_output in step_outputs
    ]

    max_density_variance = max(density_variances) if density_variances else 0.0
    max_peak_count = max(peak_counts) if peak_counts else 0
    density_mean_span = (
        max(density_means) - min(density_means)
        if density_means
        else 0.0
    )

    return {
        "field_observable": max_density_variance > 0.0,
        "peak_tracking_available": max_peak_count >= 1,
        "density_mean_changes": density_mean_span > 0.0,
        "max_density_variance": max_density_variance,
        "max_peak_count": max_peak_count,
        "density_mean_span": density_mean_span,
        "num_steps": len(step_outputs),
    }