"""
Rule-based recommendation engine: what_to_do × when_to_do.

Design principles (product + interview)
---------------------------------------
- **Inputs**: predicted emotional state, predicted intensity, stress, energy, time of day,
  and the **uncertainty layer** (downshift toward wellness-safe actions when rules say so).
- **Priority**: concrete signal-based branches (stress, energy, state) run **before** the
  uncertainty fallback so medium-confidence rows still get diverse, appropriate actions.

Action vocabulary (stable snake_case for CSV)
----------------------------------------------
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# --- Assignment-compliant timing labels (CSV `when_to_do` only) ---
VALID_TIMINGS = {"now", "within_15_min", "later_today", "tonight", "tomorrow_morning"}

# --- Actions (assignment-aligned) ---
BOX_BREATHING = "box_breathing"
JOURNALING = "journaling"
GROUNDING = "grounding"
DEEP_WORK = "deep_work"
REST = "rest"
LIGHT_PLANNING = "light_planning"
MOVEMENT = "movement"
PAUSE = "pause"

# --- Timing (internal buckets; exported via map_timing_label) ---
WHEN_NOW = "now"
WHEN_AFTER_BREAK = "after_break"
WHEN_EVENING = "this_evening"
WHEN_MORNING = "tomorrow_morning"
WHEN_STEADY = "when_steady"


def map_timing_label(raw_timing: str, time_of_day: str) -> str:
    """
    Map internal timing buckets to assignment-compliant labels.

    Internal rules still use :data:`WHEN_*` constants; this layer is the only
    export surface for CSV consumers.
    """
    raw = (raw_timing or "").strip()
    tod = (time_of_day or "").strip().lower()

    if raw in VALID_TIMINGS:
        return raw

    internal_to_canonical = {
        WHEN_AFTER_BREAK: "within_15_min",
        WHEN_STEADY: "within_15_min",
        WHEN_EVENING: "tonight",
        "later": "later_today",
        "soon": "within_15_min",
    }
    if raw in internal_to_canonical:
        mapped = internal_to_canonical[raw]
        logger.info("Mapped timing '%s' → '%s'", raw, mapped)
        return mapped

    fallback = "tonight" if tod == "night" else "later_today"
    logger.info("Mapped timing '%s' → '%s'", raw, fallback)
    return fallback


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


def _optional_float(row: dict[str, Any], name: str) -> Optional[float]:
    """None if missing/NaN — avoids treating imputed defaults as real survey values."""
    v = row.get(name)
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _recommend_raw(
    predicted_state: str,
    predicted_intensity: int,
    uncertain_flag: int,
    confidence: float,
    time_of_day: Optional[str],
    row: dict[str, Any],
) -> tuple[str, str]:
    """
    Core rule engine: returns (action, raw_timing).

    Order: (1) high stress (2) low energy (3) high energy + calm/focused
    (4) restless / overwhelmed (5) mixed / reflective (6) moderate stress
    (7) remaining state paths (8) uncertainty fallback (9) default.
    """
    _ = confidence  # reserved for future gating; uncertainty uses uncertain_flag
    stress = _get_float(row, "stress_level")
    energy = _get_float(row, "energy_level")
    stress_known = _optional_float(row, "stress_level")
    tod = (time_of_day or "").strip().lower()
    evening_night = tod in ("evening", "night")
    uncertain = uncertain_flag == 1

    # --- 1. High stress (>= 4) ---
    if stress >= 4.0:
        if energy <= 2.0:
            return REST, WHEN_EVENING if evening_night else WHEN_AFTER_BREAK
        if predicted_state == "overwhelmed":
            return LIGHT_PLANNING, WHEN_NOW
        return GROUNDING, WHEN_NOW

    # --- 2. Low energy (<= 2) ---
    if energy <= 2.0:
        if evening_night:
            return REST, WHEN_EVENING
        return PAUSE, WHEN_AFTER_BREAK

    # --- 3. High energy (>= 4) + calm / focused ---
    if energy >= 4.0 and predicted_state in ("calm", "focused"):
        return DEEP_WORK, WHEN_NOW

    # --- 4. Restless / anxious (overwhelmed below high-stress branch) ---
    if predicted_state == "restless":
        return MOVEMENT, WHEN_NOW if tod != "night" else WHEN_AFTER_BREAK
    if predicted_state == "overwhelmed":
        if energy <= 2.0:
            return REST, WHEN_EVENING if evening_night else WHEN_AFTER_BREAK
        return LIGHT_PLANNING, WHEN_NOW

    # --- 5. Reflective / mixed ---
    if predicted_state in ("mixed", "reflective"):
        return JOURNALING, "within_15_min"

    # --- 6. Moderate stress (2–3), explicit field only ---
    if stress_known is not None and 2.0 <= stress_known <= 3.0:
        return GROUNDING, "within_15_min"

    # --- 7. Remaining states (neutral, calm, focused, neutral) ---
    if predicted_state == "neutral":
        return LIGHT_PLANNING, WHEN_STEADY if energy <= 2.0 else WHEN_NOW

    if predicted_state == "calm":
        if predicted_intensity <= 2 and stress < 4.0:
            return BOX_BREATHING, WHEN_MORNING if tod in ("night", "early_morning") else WHEN_NOW
        return GROUNDING, WHEN_NOW

    if predicted_state == "focused":
        if energy >= 3 and stress < 4.0 and tod not in ("night",) and predicted_intensity >= 3:
            return DEEP_WORK, WHEN_NOW
        if energy <= 2.0 or stress >= 4.0:
            return PAUSE, WHEN_AFTER_BREAK
        return LIGHT_PLANNING, WHEN_NOW

    # --- 8. Uncertainty fallback only after signal-based rules ---
    if uncertain:
        if stress >= 3.0:
            return GROUNDING, "within_15_min"
        if energy <= 2.0:
            return REST, "later_today"
        return LIGHT_PLANNING, "later_today"

    # --- 9. Default ---
    return GROUNDING, WHEN_AFTER_BREAK


def recommend(
    predicted_state: str,
    predicted_intensity: int,
    uncertain_flag: int,
    confidence: float,
    time_of_day: Optional[str],
    row: dict[str, Any],
) -> tuple[str, str]:
    """
    Deterministic rules. Returned ``when_to_do`` is always in :data:`VALID_TIMINGS`.
    """
    action, raw_timing = _recommend_raw(
        predicted_state,
        predicted_intensity,
        uncertain_flag,
        confidence,
        time_of_day,
        row,
    )
    tod_raw = row.get("time_of_day")
    if tod_raw is None or (isinstance(tod_raw, float) and np.isnan(tod_raw)):
        tod_s = ""
    else:
        tod_s = str(tod_raw).strip()
    timing = map_timing_label(raw_timing, tod_s)
    return action, timing
