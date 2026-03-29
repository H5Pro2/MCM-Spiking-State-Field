# --------------------------------------------------
# main.py
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.exp_phase_a import (
    evaluate_phase_a_results,
    run_phase_a_experiment,
)
from src.experiments.exp_phase_b import (
    evaluate_phase_b_results,
    run_phase_b_experiment,
)
from src.experiments.exp_phase_c import (
    evaluate_phase_c_results,
    run_phase_c_experiment,
)
from src.experiments.exp_phase_d import (
    evaluate_phase_d_results,
    run_phase_d_experiment,
)
from src.experiments.exp_phase_e import (
    evaluate_phase_e_results,
    run_phase_e_experiment,
)
from src.experiments.exp_phase_f import (
    evaluate_phase_f_results,
    run_phase_f_experiment,
)


# --------------------------------------------------
# Phasenwahl
# --------------------------------------------------
def run_selected_phase(phase: str) -> dict[str, object]:
    if phase == "a":
        run_data = run_phase_a_experiment()
        evaluation = evaluate_phase_a_results(run_data)
    elif phase == "b":
        run_data = run_phase_b_experiment()
        evaluation = evaluate_phase_b_results(run_data)
    elif phase == "c":
        run_data = run_phase_c_experiment()
        evaluation = evaluate_phase_c_results(run_data)
    elif phase == "d":
        run_data = run_phase_d_experiment()
        evaluation = evaluate_phase_d_results(run_data)
    elif phase == "e":
        run_data = run_phase_e_experiment()
        evaluation = evaluate_phase_e_results(run_data)
    elif phase == "f":
        run_data = run_phase_f_experiment()
        evaluation = evaluate_phase_f_results(run_data)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    return {
        "phase": phase,
        "run": run_data,
        "evaluation": evaluation,
    }
