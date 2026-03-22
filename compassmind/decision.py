"""Rule-based decision engine for what_to_do / when_to_do (safe under low confidence)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Action vocabulary (stable strings for product + CSV)
GROUNDING = "grounding_breath_and_senses"
JOURNAL = "short_reflective_journaling"
LIGHT_PLAN = "light_planning_one_next_step"
REST = "rest_or_nap"
GENTLE_MOVE = "gentle_movement_or_stretch"
FOCUS_BLOCK = "single_focus_block_low_stimulation"
ORGANIZE = "organize_environment_or_inbox_triage"

WHEN_NOW = "now_5_to_15_min"
WHEN_AFTER_BREAK = "after_a_short_break"
WHEN_EVENING = "evening_wind_down"
WHEN_MORNING = "next_morning_fresh_start"
WHEN_WHEN_READY = "when_energy_returns"


def _stress_energy(row: dict[str, Any]) -> tuple[float, float]:
    def _get(name: str, default: float = 3.0) -> float:
        v = row.get(name)
        if v is None:
            return default
        if isinstance(v, float) and np.isnan(v):
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    return _get("stress_level"), _get("energy_level")


def recommend(
    predicted_state: str,
    predicted_intensity: int,
    uncertain_flag: bool,
    confidence: float,
    time_of_day: Optional[str],
    row: dict[str, Any],
) -> tuple[str, str]:
    """
    Deterministic rules: prefer safety when uncertainty is high or intensity/stress is high.
    """
    stress, energy = _stress_energy(row)
    tod = (time_of_day or "").lower()
    hi_stress = stress >= 4.0
    low_energy = energy <= 2.0
    hi_intensity = predicted_intensity >= 4

    if uncertain_flag or confidence < 0.42:
        action = GROUNDING
        timing = WHEN_AFTER_BREAK if hi_stress else WHEN_NOW
        return action, timing

    if predicted_state == "overwhelmed" or (hi_stress and hi_intensity):
        if low_energy:
            return REST, WHEN_EVENING if tod in ("evening", "night") else WHEN_AFTER_BREAK
        return LIGHT_PLAN, WHEN_NOW

    if predicted_state == "restless":
        return GENTLE_MOVE, WHEN_NOW if tod != "night" else WHEN_AFTER_BREAK

    if predicted_state == "mixed":
        return JOURNAL, WHEN_NOW

    if predicted_state == "neutral":
        return LIGHT_PLAN, WHEN_WHEN_READY if low_energy else WHEN_NOW

    if predicted_state == "calm":
        if predicted_intensity <= 2 and not hi_stress:
            return ORGANIZE, WHEN_MORNING if tod in ("night", "early_morning") else WHEN_NOW
        return FOCUS_BLOCK, WHEN_NOW

    if predicted_state == "focused":
        return FOCUS_BLOCK, WHEN_NOW if energy >= 3 else WHEN_AFTER_BREAK

    return GROUNDING, WHEN_AFTER_BREAK
