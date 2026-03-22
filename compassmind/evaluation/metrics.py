"""Classification metrics for emotional_state (multiclass) and intensity (5-class)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_proba: np.ndarray | None = None,
    labels: list[Any] | None = None,
) -> dict[str, float | str]:
    """Standard metrics for one task; includes macro/weighted F1 and optional log-loss."""
    out: dict[str, float | str] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    try:
        out["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    except ValueError:
        out["cohen_kappa"] = float("nan")
    if y_proba is not None:
        try:
            out["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
        except Exception:
            out["log_loss"] = float("nan")
    return out


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_names: list[str] | None = None,
) -> dict:
    """Per-class precision/recall/F1 as nested dict (sklearn)."""
    return classification_report(
        y_true,
        y_pred,
        labels=None,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
