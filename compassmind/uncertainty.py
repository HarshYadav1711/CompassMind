"""
Uncertainty layer: confidence from probabilities + transparent uncertain_flag rules.

Interview-friendly summary
--------------------------
1. **confidence** — blend of how sure the state model is and how sure the intensity model is
   (weighted max calibrated probability). Higher means “more trust in the top prediction.”

2. **uncertain_flag** (0/1) — set to 1 if *any* of these holds:
   - **Low confidence** — blended confidence below a threshold (from training bundle or default).
   - **Close top classes** — top-1 vs top-2 margin on *emotional_state* is small (model is “torn”).
   - **High entropy** — state distribution is flat-ish (normalized entropy high).
   - **Weak / short text** — journal too few characters or words to support strong inference.
   - **Sparse metadata** — many structured fields missing (we can’t anchor context).
   - **Conflicting signals** — e.g. very high stress with a “calm” face hint, or very low energy
     paired with a high-arousal state at high intensity (conservative wellness trigger).

This is intentionally rule-based so you can explain trade-offs without a second black box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

# Defaults if bundle does not override (training may tune conf/entropy only).
DEFAULT_CONF_THRESH = 0.45
DEFAULT_ENTROPY_THRESH = 0.88
DEFAULT_MARGIN_THRESH = 0.12  # top1 - top2 below this => uncertain
MIN_JOURNAL_CHARS = 40
MIN_JOURNAL_WORDS = 8
MAX_MISSING_METADATA = 5  # if >= this many of 9 structured fields missing => uncertain


def max_prob_and_entropy(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise max probability and normalized entropy in [0,1]."""
    p = np.clip(proba, 1e-12, 1.0)
    ent = -(p * np.log(p)).sum(axis=1)
    max_ent = np.log(p.shape[1])
    norm_ent = ent / max_ent if max_ent > 0 else np.zeros_like(ent)
    max_p = p.max(axis=1)
    return max_p, norm_ent


def top_two_margin(proba: np.ndarray) -> np.ndarray:
    """Per row: p[1] - p[2] after sorting descending (largest gap = more decisive)."""
    if proba.shape[1] < 2:
        return np.ones(proba.shape[0])
    s = np.sort(proba, axis=1)[:, ::-1]
    return s[:, 0] - s[:, 1]


def combined_confidence(
    p_state: np.ndarray,
    p_intensity: np.ndarray,
    w_state: float = 0.72,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      max_state_prob, max_intensity_prob, blended_confidence, norm_entropy_state
    """
    ms, es = max_prob_and_entropy(p_state)
    mi, _ei = max_prob_and_entropy(p_intensity)
    conf = w_state * ms + (1.0 - w_state) * mi
    return ms, mi, conf, es


def _count_missing_metadata(row: dict[str, Any]) -> int:
    keys = [
        "ambience_type",
        "duration_min",
        "sleep_hours",
        "energy_level",
        "stress_level",
        "time_of_day",
        "previous_day_mood",
        "face_emotion_hint",
        "reflection_quality",
    ]
    n = 0
    for k in keys:
        v = row.get(k)
        if v is None:
            n += 1
            continue
        if isinstance(v, float) and (np.isnan(v) or pd.isna(v)):
            n += 1
            continue
        if isinstance(v, str) and v.strip() == "":
            n += 1
    return n


def _journal_weakness(journal_text: str) -> tuple[bool, int, int]:
    t = (journal_text or "").strip()
    nchar = len(t)
    nwords = len(t.split()) if t else 0
    weak = (nchar < MIN_JOURNAL_CHARS) or (nwords < MIN_JOURNAL_WORDS)
    return weak, nchar, nwords


def _conflicting_signals(
    row: dict[str, Any],
    predicted_state: str,
    predicted_intensity: int,
) -> bool:
    """Conservative wellness triggers when numeric hints disagree with labels or each other."""

    def _f(name: str) -> Optional[float]:
        v = row.get(name)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    stress = _f("stress_level")
    energy = _f("energy_level")

    def _norm_str(val: Any) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        return str(val).strip().lower()

    face = _norm_str(row.get("face_emotion_hint"))
    prev_mood = _norm_str(row.get("previous_day_mood"))

    # High stress but face hint says calm — often noisy OCR/metadata.
    if stress is not None and stress >= 4.5 and face == "calm_face":
        return True

    # Very low energy but model says focused at high intensity — treat as unstable.
    if energy is not None and energy <= 1.5 and predicted_state == "focused" and predicted_intensity >= 4:
        return True

    # Overwhelmed previous day + model says calm at low intensity — possible mismatch.
    if prev_mood == "overwhelmed" and predicted_state == "calm" and predicted_intensity <= 2:
        return True

    # Extreme stress vs extreme low energy simultaneously (inconsistent self-report).
    if stress is not None and energy is not None and stress >= 4.5 and energy <= 1.5:
        return True

    return False


@dataclass
class UncertaintyConfig:
    """Tunable knobs; bundle may override conf/entropy thresholds."""

    conf_thresh: float = DEFAULT_CONF_THRESH
    entropy_thresh: float = DEFAULT_ENTROPY_THRESH
    margin_thresh: float = DEFAULT_MARGIN_THRESH
    min_chars: int = MIN_JOURNAL_CHARS
    min_words: int = MIN_JOURNAL_WORDS
    max_missing_meta: int = MAX_MISSING_METADATA


def compute_uncertain_flag(
    *,
    confidence: float,
    max_state_prob: float,
    norm_entropy_state: float,
    margin_state: float,
    journal_weak: bool,
    missing_meta: int,
    conflicting: bool,
    cfg: UncertaintyConfig,
) -> int:
    """Return 1 if uncertain, else 0 (transparent OR-of-rules)."""
    if confidence < cfg.conf_thresh:
        return 1
    if norm_entropy_state > cfg.entropy_thresh:
        return 1
    if margin_state < cfg.margin_thresh:
        return 1
    if journal_weak:
        return 1
    if missing_meta >= cfg.max_missing_meta:
        return 1
    if conflicting:
        return 1
    return 0


def build_uncertainty_config(bundle: dict[str, Any]) -> UncertaintyConfig:
    """Merge training-tuned thresholds with default margin/text/metadata rules."""
    return UncertaintyConfig(
        conf_thresh=float(bundle.get("conf_thresh", DEFAULT_CONF_THRESH)),
        entropy_thresh=float(bundle.get("ent_thresh", DEFAULT_ENTROPY_THRESH)),
        margin_thresh=DEFAULT_MARGIN_THRESH,
        min_chars=MIN_JOURNAL_CHARS,
        min_words=MIN_JOURNAL_WORDS,
        max_missing_meta=MAX_MISSING_METADATA,
    )


def combined_scores(
    p_state: np.ndarray,
    p_intensity: np.ndarray,
    w_state: float = 0.72,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Alias used by training (`train_eval`): same return shape as historical API."""
    return combined_confidence(p_state, p_intensity, w_state=w_state)


def uncertain_mask(
    max_state_prob: np.ndarray,
    norm_ent_state: np.ndarray,
    conf_thresh: float,
    ent_thresh: float,
) -> np.ndarray:
    """
    Training-time tuning only: threshold max state probability + normalized entropy.
    Inference uses `compute_uncertain_flag` per row with additional rules.
    """
    return (max_state_prob < conf_thresh) | (norm_ent_state > ent_thresh)
