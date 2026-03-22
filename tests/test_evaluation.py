"""Lightweight tests for evaluation metrics (no full training run)."""

import numpy as np

from compassmind.evaluation.metrics import classification_metrics


def test_classification_metrics_basic():
    y_t = np.array([0, 1, 2, 0])
    y_p = np.array([0, 1, 1, 0])
    m = classification_metrics(y_t, y_p)
    assert m["accuracy"] == 0.75
    assert "macro_f1" in m
