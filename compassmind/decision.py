"""
Rule-based recommendation engine: what_to_do × when_to_do.

Design principles (product + interview)
---------------------------------------
- **Inputs**: predicted emotional state, predicted intensity, stress, energy, time of day,
  and the **uncertainty layer** (not just accuracy — we downshift toward wellness-safe actions).
- **Defaults**: if `uncertain_flag == 1` or confidence is low, prefer *grounding*, *breathing*,
  *pause*, *journaling*, *rest* — not aggressive productivity.
- **Conflict policy**: when stress/energy disagree with a “push hard” interpretation, choose
  conservative regulation skills first.

Action vocabulary (stable snake_case for CSV)
----------------------------------------------
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# --- Actions (assignment-aligned) ---
BOX_BREATHING = "box_breathing"
JOURNALING = "journaling"
GROUNDING = "grounding"
DEEP_WORK = "deep_work"
REST = "rest"
LIGHT_PLANNING = "light_planning"
MOVEMENT = "movement"
PAUSE = "pause"

# --- Timing (transparent buckets) ---
WHEN_NOW = "now"
WHEN_AFTER_BREAK = "after_break"
WHEN_EVENING = "this_evening"
WHEN_MORNING = "tomorrow_morning"
WHEN_STEADY = "when_steady"


def _get_float(row: dict[str, Any], name: str, default: float = 3.0) -> float:
    v = row.get(name)
    if v is None:
        return default
    if isinstance(v, float) and np.isnan(v):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def recommend(
    predicted_state: str,
    predicted_intensity: int,
    uncertain_flag: int,
    confidence: float,
    time_of_day: Optional[str],
    row: dict[str, Any],
) -> tuple[str, str]:
    """
    Deterministic rules. `uncertain_flag` is 0/1 from the uncertainty layer.

    Conservative path: uncertainty → grounding / breathing / pause / journaling.
    Productive path: only when confidence is adequate and signals align.
    """
    stress = _get_float(row, "stress_level")
    energy = _get_float(row, "energy_level")
    tod = (time_of_day or "").strip().lower()

    hi_stress = stress >= 4.0
    low_energy = energy <= 2.0
    hi_intensity = predicted_intensity >= 4
    low_conf = confidence < 0.42
    uncertain = uncertain_flag == 1

    # --- Safe defaults: uncertainty, low confidence, or conflicting high load ---
    if uncertain or low_conf:
        if hi_stress or hi_intensity:
            return BOX_BREATHING, WHEN_AFTER_BREAK
        return GROUNDING, WHEN_NOW

    # --- Primary mapping (wellness-biased) ---
    if predicted_state == "overwhelmed" or (hi_stress and hi_intensity):
        if low_energy:
            return REST, WHEN_EVENING if tod in ("evening", "night") else WHEN_AFTER_BREAK
        return LIGHT_PLANNING, WHEN_NOW

    if predicted_state == "restless":
        return MOVEMENT, WHEN_NOW if tod != "night" else WHEN_AFTER_BREAK

    if predicted_state == "mixed":
        return JOURNALING, WHEN_NOW

    if predicted_state == "neutral":
        return LIGHT_PLANNING, WHEN_STEADY if low_energy else WHEN_NOW

    if predicted_state == "calm":
        if predicted_intensity <= 2 and not hi_stress:
            return BOX_BREATHING, WHEN_MORNING if tod in ("night", "early_morning") else WHEN_NOW
        return GROUNDING, WHEN_NOW

    if predicted_state == "focused":
        # Deep work only with sufficient energy and not at night by default
        if energy >= 3 and not hi_stress and tod not in ("night",) and predicted_intensity >= 3:
            return DEEP_WORK, WHEN_NOW
        if low_energy or hi_stress:
            return PAUSE, WHEN_AFTER_BREAK
        return LIGHT_PLANNING, WHEN_NOW

    return GROUNDING, WHEN_AFTER_BREAK
