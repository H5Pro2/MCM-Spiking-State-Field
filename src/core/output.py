# --------------------------------------------------
# src/core/output.py
# --------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


# --------------------------------------------------
# Labels
# --------------------------------------------------
def build_readout_labels(
    mcm_state: float,
    neural_activity: float,
) -> list[str]:
    labels: list[str] = []

    if mcm_state > 0.5:
        labels.append("positive_shift")
    elif mcm_state < -0.5:
        labels.append("negative_shift")
    else:
        labels.append("centered")

    if neural_activity >= 0.5:
        labels.append("high_activity")
    else:
        labels.append("low_activity")

    return labels


# --------------------------------------------------
# Output-State
# --------------------------------------------------
def build_output_state(
    neural_state: Mapping[str, object],
    mcm_state: float,
) -> dict[str, object]:
    activity = float(neural_state.get("activity", 0.0))
    spike = float(neural_state.get("spike", 0.0))

    return {
        "mcm_state": float(mcm_state),
        "neural_activity": activity,
        "spike": spike,
        "labels": build_readout_labels(
            mcm_state=float(mcm_state),
            neural_activity=activity,
        ),
    }


# --------------------------------------------------
# Serialisierung
# --------------------------------------------------
def serialize_output_state(output_state: Mapping[str, object]) -> str:
    return json.dumps(dict(output_state), ensure_ascii=False, sort_keys=True)


# --------------------------------------------------
# Export
# --------------------------------------------------
def export_output_snapshot(output_state: Mapping[str, object], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        serialize_output_state(output_state),
        encoding="utf-8",
    )
    return output_path