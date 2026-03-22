"""
Evaluation and analysis: holdout metrics, ablation, feature attribution, robustness, reporting.

Run: ``python -m compassmind.evaluation`` (writes ``artifacts/evaluation_report.json`` and refreshes docs).
"""

from compassmind.evaluation.holdout import HoldoutResult, evaluate_holdout
from compassmind.evaluation.metrics import classification_metrics
from compassmind.evaluation.run import build_report, write_markdown

__all__ = [
    "evaluate_holdout",
    "HoldoutResult",
    "classification_metrics",
    "build_report",
    "write_markdown",
]
