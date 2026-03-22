"""Uncertainty layer unit tests."""

import numpy as np

from compassmind.uncertainty import (
    UncertaintyConfig,
    compute_uncertain_flag,
    top_two_margin,
)


def test_close_top_classes_low_margin_flags_uncertain():
    p = np.array([[0.34, 0.33, 0.33, 0.0, 0.0, 0.0]])
    assert float(top_two_margin(p)[0]) < 0.12


def test_compute_uncertain_flag_or_logic():
    cfg = UncertaintyConfig(conf_thresh=0.5, entropy_thresh=0.9, margin_thresh=0.15)
    assert (
        compute_uncertain_flag(
            confidence=0.9,
            max_state_prob=0.9,
            norm_entropy_state=0.1,
            margin_state=0.4,
            journal_weak=False,
            missing_meta=0,
            conflicting=False,
            cfg=cfg,
        )
        == 0
    )
    assert (
        compute_uncertain_flag(
            confidence=0.2,
            max_state_prob=0.2,
            norm_entropy_state=0.1,
            margin_state=0.4,
            journal_weak=False,
            missing_meta=0,
            conflicting=False,
            cfg=cfg,
        )
        == 1
    )
