# --------------------------------------------------
# src/experiments/exp_phase_e.py
# --------------------------------------------------
from __future__ import annotations

from src.core.context import initialize_context_state, update_context_state
from src.core.field_density import build_field_axis, compute_density_variance, reconstruct_density
from src.core.mcm_state import compute_state_velocity, initialize_mcm_state, update_mcm_state
from src.core.neural_core import build_neural_core, decode_neural_state, step_neural_core
from src.core.perception import apply_input_gain, encode_perception
from src.core.reflection import build_reflection_state
from src.core.self_state import build_self_state, build_self_state_labels


def build_phase_e_inputs() -> list[dict[str, float]]:
    return [
        {"valence": 0.9, "novelty": 0.5, "relevance": 0.7, "uncertainty": 0.2, "social_salience": 0.0},
        {"valence": 0.9, "novelty": 0.5, "relevance": 0.7, "uncertainty": 0.2, "social_salience": 0.0},
        {"valence": -0.9, "novelty": 0.6, "relevance": 0.7, "uncertainty": 0.3, "social_salience": 0.1},
        {"valence": 0.0, "novelty": 0.0, "relevance": 0.0, "uncertainty": 0.0, "social_salience": 0.0},
        {"valence": -0.7, "novelty": 0.7, "relevance": 0.8, "uncertainty": 0.4, "social_salience": 0.1},
    ]


def run_phase_e_experiment() -> dict[str, object]:
    dt = 0.1
    field_axis = build_field_axis()
    preferred_positions = (-2.0, -1.0, 0.0, 1.0, 2.0)

    neural_state = build_neural_core()
    x_prev = initialize_mcm_state()
    context_state = initialize_context_state()
    previous_self_state: dict[str, float] = {"center_distance": 0.0}

    step_outputs: list[dict[str, object]] = []

    for step_index, raw_input in enumerate(build_phase_e_inputs()):
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
        density = reconstruct_density(
            neural_activity=(
                max(0.0, activity + abs(raw_input["valence"]) * 0.5),
                max(0.0, activity + raw_input["novelty"] * 0.4),
                max(0.0, activity + raw_input["relevance"] * 0.3),
                max(0.0, activity + raw_input["uncertainty"] * 0.2),
                max(0.0, activity + raw_input["social_salience"] * 0.1),
            ),
            preferred_positions=preferred_positions,
            field_axis=field_axis,
        )
        density_variance = compute_density_variance(density=density, field_axis=field_axis)

        self_state = build_self_state(
            x_value=x_curr,
            velocity=velocity,
            density_variance=density_variance,
            cluster_stability=max(0.0, 1.0 - min(1.0, density_variance * 0.3)),
        )
        self_labels = build_self_state_labels(self_state)

        context_state = update_context_state(
            previous_context=context_state,
            x_value=x_curr,
            density_variance=density_variance,
            cluster_id=-1.0,
            input_energy=sum(abs(value) for value in input_vector),
            self_state=self_state,
        )

        reflection_state = build_reflection_state(
            x_value=x_curr,
            velocity=velocity,
            context_state=context_state,
            previous_self_state=previous_self_state,
            matched_cluster_ids=(),
        )

        step_outputs.append(
            {
                "step": step_index,
                "x": x_curr,
                "velocity": velocity,
                "self_state": self_state,
                "self_labels": self_labels,
                "reflection": reflection_state,
                "context": dict(context_state),
            }
        )

        previous_self_state = dict(self_state)
        x_prev = x_curr

    return {
        "dt": dt,
        "steps": step_outputs,
    }


def evaluate_phase_e_results(run_data: dict[str, object]) -> dict[str, object]:
    steps = list(run_data["steps"])

    loop_risk_count = sum(1 for step in steps if float(step["reflection"]["loop_risk"]) > 0.0)
    drift_detected_count = sum(1 for step in steps if float(step["reflection"]["drift_detected"]) > 0.0)
    return_detected_count = sum(1 for step in steps if float(step["reflection"]["return_toward_center"]) > 0.0)
    has_labels = any(len(list(step["self_labels"])) > 0 for step in steps)

    return {
        "self_state_available": len(steps) > 0,
        "self_labels_available": has_labels,
        "drift_detected_count": drift_detected_count,
        "return_detected_count": return_detected_count,
        "loop_risk_count": loop_risk_count,
        "num_steps": len(steps),
    }
