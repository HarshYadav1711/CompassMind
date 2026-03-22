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

# --- Timing (transparent buckets) ---
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


def _recommend_raw(
    predicted_state: str,
    predicted_intensity: int,
    uncertain_flag: int,
    confidence: float,
    time_of_day: Optional[str],
    row: dict[str, Any],
) -> tuple[str, str]:
    """
    Core rule engine: returns (action, raw_timing) using internal WHEN_* labels.
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

    Returned ``when_to_do`` is always in :data:`VALID_TIMINGS` (see :func:`map_timing_label`).
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
