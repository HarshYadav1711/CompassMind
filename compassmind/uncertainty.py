"""Uncertainty from predicted probabilities (calibrated) + normalized entropy."""

from __future__ import annotations

import numpy as np


def max_prob_and_entropy(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise max probability and normalized entropy in [0,1]."""
    p = np.clip(proba, 1e-12, 1.0)
    ent = -(p * np.log(p)).sum(axis=1)
    max_ent = np.log(p.shape[1])
    norm_ent = ent / max_ent if max_ent > 0 else np.zeros_like(ent)
    max_p = p.max(axis=1)
    return max_p, norm_ent


def combined_scores(
    p_state: np.ndarray,
    p_intensity: np.ndarray,
    w_state: float = 0.72,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """max_state, max_intensity, weighted confidence, norm_entropy_state."""
    ms, es = max_prob_and_entropy(p_state)
    mi, _ei = max_prob_and_entropy(p_intensity)
    # Confidence is uncertainty-aware but dominated by emotional_state (primary decision).
    conf = w_state * ms + (1.0 - w_state) * mi
    return ms, mi, conf, es


def uncertain_mask(
    max_state_prob: np.ndarray,
    norm_ent_state: np.ndarray,
    conf_thresh: float = 0.48,
    ent_thresh: float = 0.82,
) -> np.ndarray:
    """Primary uncertainty uses state distribution (calibrated), not the blended confidence."""
    return (max_state_prob < conf_thresh) | (norm_ent_state > ent_thresh)
