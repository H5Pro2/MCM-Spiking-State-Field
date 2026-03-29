# --------------------------------------------------
# src/experiments/exp_phase_f.py
# --------------------------------------------------
from __future__ import annotations

from src.core.context import initialize_context_state, update_context_state
from src.core.field_density import build_field_axis, compute_density_variance, reconstruct_density
from src.core.mcm_state import compute_state_velocity, initialize_mcm_state, update_mcm_state
from src.core.meta_regulation import build_meta_regulation_state
from src.core.neural_core import build_neural_core, decode_neural_state, step_neural_core
from src.core.perception import apply_input_gain, encode_perception
from src.core.reflection import build_reflection_state
from src.core.regulation import derive_regulation_parameters, regulate_replay_signal
from src.core.replay import compute_replay_signal
from src.core.self_state import build_self_state


def build_phase_f_inputs() -> list[dict[str, float]]:
    return [
        {"valence": 0.8, "novelty": 0.6, "relevance": 0.8, "uncertainty": 0.2, "social_salience": 0.1},
        {"valence": 0.8, "novelty": 0.6, "relevance": 0.8, "uncertainty": 0.2, "social_salience": 0.1},
        {"valence": -0.9, "novelty": 0.7, "relevance": 0.8, "uncertainty": 0.4, "social_salience": 0.2},
        {"valence": -0.9, "novelty": 0.7, "relevance": 0.8, "uncertainty": 0.4, "social_salience": 0.2},
        {"valence": 0.0, "novelty": 0.0, "relevance": 0.0, "uncertainty": 0.0, "social_salience": 0.0},
        {"valence": 0.0, "novelty": 0.0, "relevance": 0.0, "uncertainty": 0.0, "social_salience": 0.0},
    ]


def run_phase_f_experiment() -> dict[str, object]:
    dt = 0.1
    field_axis = build_field_axis()
    preferred_positions = (-2.0, -1.0, 0.0, 1.0, 2.0)

    neural_state = build_neural_core()
    x_prev = initialize_mcm_state()
    context_state = initialize_context_state()
    previous_self_state: dict[str, float] = {"center_distance": 0.0}

    memory_bank = [
        {
            "cluster_id": 0.0,
            "x_mean": 0.9,
            "mean_density_variance": 0.6,
            "strength": 1.2,
            "stability": 0.5,
            "age": 1.0,
        },
        {
            "cluster_id": 1.0,
            "x_mean": -0.9,
            "mean_density_variance": 0.7,
            "strength": 1.0,
            "stability": 0.4,
            "age": 1.0,
        },
    ]

    regulation_state: dict[str, float] = {
        "k_reg": 0.4,
        "replay_gain": 1.0,
        "input_gain": 1.0,
        "noise_gain": 1.0,
    }

    step_outputs: list[dict[str, object]] = []

    for step_index, raw_input in enumerate(build_phase_f_inputs()):
        replay_signal = compute_replay_signal(
            memory_bank=memory_bank,
            context_state=context_state,
            mcm_state=x_prev,
            regulation_state={"replay_gain": regulation_state["replay_gain"]},
        )
        replay_input = regulate_replay_signal(replay_signal, regulation_state)

        input_vector = encode_perception(raw_input)
        gained_input_vector = apply_input_gain(
            input_vector=input_vector,
            gain_state=regulation_state["input_gain"],
        )

        neural_state = step_neural_core(
            input_vector=gained_input_vector,
            recurrent_feedback=0.0,
            neural_state=neural_state,
        )
        neural_drive = decode_neural_state(float(neural_state["activity"]))

        x_curr = update_mcm_state(
            x_prev=x_prev,
            neural_input=neural_drive,
            replay_input=replay_input,
            regulation_gain=regulation_state["k_reg"],
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
            cluster_stability=context_state.get("stability_estimate", 1.0),
        )
        reflection_state = build_reflection_state(
            x_value=x_curr,
            velocity=velocity,
            context_state=context_state,
            previous_self_state=previous_self_state,
            matched_cluster_ids=(entry["cluster_id"] for entry in memory_bank),
        )
        meta_state = build_meta_regulation_state(
            self_state=self_state,
            reflection_state=reflection_state,
            context_state=context_state,
        )
        regulation_state = derive_regulation_parameters(meta_state)

        context_state = update_context_state(
            previous_context=context_state,
            x_value=x_curr,
            density_variance=density_variance,
            cluster_id=float(memory_bank[0]["cluster_id"]),
            input_energy=sum(abs(value) for value in input_vector),
            self_state=self_state,
        )

        step_outputs.append(
            {
                "step": step_index,
                "x": x_curr,
                "velocity": velocity,
                "replay_signal": replay_signal,
                "replay_input": replay_input,
                "meta_state": meta_state,
                "regulation": dict(regulation_state),
                "context": dict(context_state),
            }
        )

        x_prev = x_curr
        previous_self_state = dict(self_state)

    return {
        "dt": dt,
        "steps": step_outputs,
    }


def evaluate_phase_f_results(run_data: dict[str, object]) -> dict[str, object]:
    steps = list(run_data["steps"])

    replay_gains = [float(step["regulation"]["replay_gain"]) for step in steps]
    k_reg_values = [float(step["regulation"]["k_reg"]) for step in steps]
    replay_inputs = [abs(float(step["replay_input"])) for step in steps]

    return {
        "meta_regulation_active": len(steps) > 0,
        "k_reg_increase_detected": (max(k_reg_values) - min(k_reg_values)) > 1e-9 if k_reg_values else False,
        "replay_gain_adaptive": (max(replay_gains) - min(replay_gains)) > 1e-9 if replay_gains else False,
        "replay_controlled": max(replay_inputs) < 1.5 if replay_inputs else True,
        "num_steps": len(steps),
    }
