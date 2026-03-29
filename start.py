# --------------------------------------------------
# start.py
# --------------------------------------------------
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.debug_reader import dbr_debug
from main import run_selected_phase


# --------------------------------------------------
# Test-Targets
# --------------------------------------------------
TEST_TARGETS: dict[str, list[str]] = {
    "all": ["-q"],
    "experiments": ["-q", "tests/test_experiments_pipeline.py"],
    "core": [
        "-q",
        "tests/test_mcm_state.py",
        "tests/test_field_density.py",
        "tests/test_clustering.py",
    ],
    "regulation": ["-q", "tests/test_context.py", "tests/test_regulation.py"],
}


# --------------------------------------------------
# CLI-Parser
# --------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Zentrale Startverwaltung fuer Tests und MCM-Spiking-State-Field-Phasen."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("tests", "phase"),
        default="tests",
        help="tests = pytest starten, phase = Experimentphase laufen lassen",
    )
    parser.add_argument(
        "--target",
        choices=tuple(TEST_TARGETS.keys()),
        default="all",
        help="Welches Testpaket ausgefuehrt werden soll (nur fuer --mode tests).",
    )
    parser.add_argument(
        "--phase",
        choices=("a", "b", "c", "d", "e", "f"),
        default="a",
        help="Welche Phase ausgefuehrt werden soll (nur fuer --mode phase).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Bei --mode phase nur die Auswertung ausgeben.",
    )
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="Kein Konsolenoutput; nur Debug-Log-Datei schreiben.",
    )
    return parser


# --------------------------------------------------
# Test-Runner
# --------------------------------------------------
def run_tests(target: str) -> int:
    pytest_args = TEST_TARGETS[target]
    return int(pytest.main(pytest_args))


# --------------------------------------------------
# Phase-Runner
# --------------------------------------------------
def run_phase(
    phase: str,
    compact: bool,
    no_stdout: bool,
) -> int:
    result = run_selected_phase(phase)
    payload_object = result["evaluation"] if compact else result
    payload_text = json.dumps(payload_object, indent=2, ensure_ascii=False)

    dbr_debug(payload_text, "experiment_results.txt")

    #if not no_stdout:
        #print(payload_text)

    return 0


# --------------------------------------------------
# Startpunkt
# --------------------------------------------------
def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print(f"Projektpfad: {PROJECT_ROOT}")

    if args.mode == "tests":
        return run_tests(args.target)

    return run_phase(
        phase=args.phase,
        compact=args.compact,
        no_stdout=args.no_stdout,
    )


# --------------------------------------------------
# Direkter Aufruf
# --------------------------------------------------
if __name__ == "__main__":
    raise SystemExit(main())