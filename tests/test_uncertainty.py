"""Uncertainty layer unit tests."""

import numpy as np

from compassmind.uncertainty import (
    UncertaintyConfig,
    _conflicting_signals,
    compute_uncertain_flag,
    top_two_margin,
)


def test_close_top_classes_low_margin_flags_uncertain():
    p = np.array([[0.34, 0.33, 0.33, 0.0, 0.0, 0.0]])
    assert float(top_two_margin(p)[0]) < 0.12


def test_conflicting_signals_handles_nan_float_metadata():
    row = {
        "stress_level": 2.0,
        "energy_level": 3.0,
        "face_emotion_hint": float("nan"),
        "previous_day_mood": float("nan"),
    }
    assert _conflicting_signals(row, "calm", 2) is False


def test_compute_uncertain_flag_low_confidence():
    cfg = UncertaintyConfig()
    assert (
        compute_uncertain_flag(
            confidence=0.9,
            max_state_prob=0.9,
            norm_entropy_state=0.95,
            margin_state=0.4,
            journal_weak=False,
            missing_meta=8,
            conflicting=False,
            cfg=cfg,
        )
        == 0
    )
    assert (
        compute_uncertain_flag(
            confidence=0.35,
            max_state_prob=0.35,
            norm_entropy_state=0.1,
            margin_state=0.4,
            journal_weak=False,
            missing_meta=0,
            conflicting=False,
            cfg=cfg,
        )
        == 1
    )


def test_compute_uncertain_flag_gap_and_medium_confidence():
    cfg = UncertaintyConfig()
    assert (
        compute_uncertain_flag(
            confidence=0.5,
            max_state_prob=0.3,
            norm_entropy_state=0.2,
            margin_state=0.04,
            journal_weak=False,
            missing_meta=0,
            conflicting=False,
            cfg=cfg,
        )
        == 1
    )
    assert (
        compute_uncertain_flag(
            confidence=0.56,
            max_state_prob=0.3,
            norm_entropy_state=0.2,
            margin_state=0.04,
            journal_weak=False,
            missing_meta=0,
            conflicting=False,
            cfg=cfg,
        )
        == 0
    )


def test_journal_weak_alone_not_uncertain():
    cfg = UncertaintyConfig()
    assert (
        compute_uncertain_flag(
            confidence=0.85,
            max_state_prob=0.8,
            norm_entropy_state=0.3,
            margin_state=0.3,
            journal_weak=True,
            missing_meta=0,
            conflicting=False,
            cfg=cfg,
        )
        == 0
    )
