# --------------------------------------------------
# src/experiments/exp_phase_a.py
# --------------------------------------------------

from __future__ import annotations

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
# Phase-A-Inputs
# --------------------------------------------------
def build_phase_a_inputs() -> list[dict[str, float]]:
    return [
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
            "relevance": 0.6,
            "uncertainty": 0.1,
            "social_salience": 0.0,
        },
        {
            "valence": 0.8,
            "novelty": 0.4,
            "relevance": 0.6,
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
        {
            "valence": 0.0,
            "novelty": 0.0,
            "relevance": 0.0,
            "uncertainty": 0.0,
            "social_salience": 0.0,
        },
    ]


# --------------------------------------------------
# Phase-A-Experiment
# --------------------------------------------------
def run_phase_a_experiment() -> dict[str, object]:
    dt = 0.1
    regulation_gain = 0.4
    replay_input = 0.0
    recurrent_feedback = 0.0
    noise = 0.0
    input_gain = 1.0

    neural_state = build_neural_core()
    x_prev = initialize_mcm_state()
    step_outputs: list[dict[str, object]] = []

    for step_index, raw_input in enumerate(build_phase_a_inputs()):
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

        output_state = build_output_state(
            neural_state=neural_state,
            mcm_state=x_curr,
        )
        output_state["step"] = step_index
        output_state["velocity"] = velocity
        output_state["raw_input"] = dict(raw_input)

        step_outputs.append(output_state)
        x_prev = x_curr

    return {
        "dt": dt,
        "regulation_gain": regulation_gain,
        "steps": step_outputs,
    }


# --------------------------------------------------
# Phase-A-Auswertung
# --------------------------------------------------
def evaluate_phase_a_results(run_data: dict[str, object]) -> dict[str, object]:
    step_outputs = list(run_data["steps"])
    x_series = [float(step_output["mcm_state"]) for step_output in step_outputs]

    max_abs_x = max(abs(x_value) for x_value in x_series) if x_series else 0.0
    final_abs_x = abs(x_series[-1]) if x_series else 0.0

    return {
        "x_shift_detected": max_abs_x > 0.05,
        "returns_toward_center": final_abs_x < max_abs_x,
        "max_abs_x": max_abs_x,
        "final_x": x_series[-1] if x_series else 0.0,
        "num_steps": len(step_outputs),
    }