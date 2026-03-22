"""Feature attribution for sparse linear heads: text (word/char) vs metadata blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class BlockImportance:
    word_l2: float
    char_l2: float
    meta_l2: float
    word_share: float
    char_share: float
    meta_share: float
    n_word: int
    n_char: int
    n_meta: int


def _coef_feature_norm(clf: CalibratedClassifierCV) -> np.ndarray:
    """Per-feature L2 norm of coefficient matrix across classes (multiclass logistic)."""
    cal = clf.calibrated_classifiers_[0]
    est = getattr(cal, "estimator", getattr(cal, "base_estimator", None))
    if est is None:
        raise ValueError("Cannot extract base estimator from CalibratedClassifierCV")
    c = est.coef_
    if c.ndim == 1:
        return np.abs(c)
    return np.linalg.norm(c, axis=0)


def block_importance(
    clf: CalibratedClassifierCV,
    *,
    n_word: int,
    n_char: int,
    use_metadata: bool,
) -> BlockImportance:
    w = _coef_feature_norm(clf)
    n_text = n_word + n_char
    n_meta = int(w.shape[0] - n_text) if use_metadata else 0
    if w.shape[0] != n_text + n_meta:
        raise ValueError(
            f"Feature dim mismatch: coef {w.shape[0]} vs word+char+meta {n_word}+{n_char}+{n_meta}"
        )
    w_word = float(np.linalg.norm(w[:n_word]))
    w_char = float(np.linalg.norm(w[n_word : n_word + n_char]))
    w_meta = float(np.linalg.norm(w[n_word + n_char :])) if use_metadata else 0.0
    tot = w_word + w_char + w_meta + 1e-12
    return BlockImportance(
        word_l2=w_word,
        char_l2=w_char,
        meta_l2=w_meta,
        word_share=w_word / tot,
        char_share=w_char / tot,
        meta_share=w_meta / tot,
        n_word=n_word,
        n_char=n_char,
        n_meta=n_meta,
    )


def top_text_features(
    clf: CalibratedClassifierCV,
    feature_names: list[str],
    *,
    top_k: int = 15,
    n_text_features: int | None = None,
) -> list[tuple[str, float]]:
    """
    Top tokens/ngrams by coefficient norm.

    If ``n_text_features`` is set, only the first *n* coefficients are used
    (word + char TF-IDF block when metadata is appended after text).
    """
    w = _coef_feature_norm(clf)
    if n_text_features is not None:
        w = w[:n_text_features]
    if len(feature_names) != w.shape[0]:
        raise ValueError(
            f"feature_names length {len(feature_names)} must match coef slice {w.shape[0]}"
        )
    idx = np.argsort(-w)[:top_k]
    return [(feature_names[int(i)], float(w[int(i)])) for i in idx]
