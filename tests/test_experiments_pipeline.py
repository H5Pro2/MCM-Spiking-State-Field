# --------------------------------------------------
# tests/test_experiments_pipeline.py
# --------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.exp_phase_d import evaluate_phase_d_results, run_phase_d_experiment
from src.experiments.exp_phase_e import evaluate_phase_e_results, run_phase_e_experiment
from src.experiments.exp_phase_f import evaluate_phase_f_results, run_phase_f_experiment


def test_phase_d_context_pipeline_runs_and_learns_transitions() -> None:
    run_data = run_phase_d_experiment()
    evaluation = evaluate_phase_d_results(run_data)

    assert evaluation["context_built"] is True
    assert evaluation["context_learning_active"] is True
    assert evaluation["transition_count"] > 0


def test_phase_e_reflection_and_self_state_pipeline_runs() -> None:
    run_data = run_phase_e_experiment()
    evaluation = evaluate_phase_e_results(run_data)

    assert evaluation["self_state_available"] is True
    assert evaluation["self_labels_available"] is True
    assert evaluation["num_steps"] > 0


def test_phase_f_meta_regulation_pipeline_runs() -> None:
    run_data = run_phase_f_experiment()
    evaluation = evaluate_phase_f_results(run_data)

    assert evaluation["meta_regulation_active"] is True
    assert evaluation["replay_controlled"] is True
    assert evaluation["num_steps"] > 0
