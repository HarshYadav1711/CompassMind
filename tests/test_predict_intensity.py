"""Intensity from probability expectation + deterministic signal nudges."""

import numpy as np

from compassmind.predict import (
    _adjust_intensity_signals,
    _expected_intensity_from_proba,
)


def test_expected_intensity_spreads_vs_argmax_peak():
    class_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pi = np.array([[0.05, 0.15, 0.5, 0.25, 0.05]])
    out = _expected_intensity_from_proba(pi, class_values)
    assert out[0] == 3


def test_adjust_intensity_stress_energy_confidence():
    assert _adjust_intensity_signals(2, 4.5, 3.0, 0.9) == 3  # stress +1
    assert _adjust_intensity_signals(4, 2.0, 1.5, 0.9) == 3  # energy -1
    assert _adjust_intensity_signals(5, float("nan"), float("nan"), 0.40) == 4  # pull toward 3
    assert _adjust_intensity_signals(1, float("nan"), float("nan"), 0.40) == 2
