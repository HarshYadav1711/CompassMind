"""Single stratified holdout evaluation (matches training split semantics)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from compassmind.features import (
    FeatureConfig,
    MetadataEncoder,
    combine_features,
    fit_transform_text_features,
    transform_text_features,
)
from compassmind.train_eval import (
    Backend,
    build_calibrated_intensity_clf,
    build_calibrated_state_clf,
    prepare_labels,
)


@dataclass
class HoldoutResult:
    use_metadata: bool
    clf_state: CalibratedClassifierCV
    clf_intensity: CalibratedClassifierCV
    wv: Any
    cv_vec: Any
    meta_enc: MetadataEncoder | None
    le_s: Any
    le_i: Any
    X_val: Any
    val_df: pd.DataFrame
    ys_va: np.ndarray
    yi_va: np.ndarray
    pred_state: np.ndarray
    pred_intensity: np.ndarray
    proba_state: np.ndarray
    proba_intensity: np.ndarray
    n_word: int
    n_char: int


def _fit_pair(
    X_tr: Any,
    ys_tr: np.ndarray,
    yi_tr: np.ndarray,
    backend: Backend,
    random_state: int,
    calibration_cv: int,
) -> tuple[CalibratedClassifierCV, CalibratedClassifierCV]:
    clf_s = build_calibrated_state_clf(backend, random_state, calibration_cv=calibration_cv)
    clf_i = build_calibrated_intensity_clf(backend, random_state, calibration_cv=calibration_cv)
    clf_s.fit(X_tr, ys_tr)
    clf_i.fit(X_tr, yi_tr)
    return clf_s, clf_i


def evaluate_holdout(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    *,
    use_metadata: bool,
    random_state: int = 42,
    test_size: float = 0.15,
    backend: Backend = "logistic",
    calibration_cv: int = 3,
) -> HoldoutResult:
    """
    Fit on train split only; return models and validation predictions for analysis.
    Does not refit on full data (unlike ``train_bundle``).
    """
    y_s, y_i, le_s, le_i = prepare_labels(df)

    X_train, X_val, ys_tr, ys_va, yi_tr, yi_va = train_test_split(
        df,
        y_s,
        y_i,
        test_size=test_size,
        random_state=random_state,
        stratify=y_s,
    )

    X_text_tr, wv, cv_vec = fit_transform_text_features(X_train, cfg)
    X_text_va = transform_text_features(X_val, wv, cv_vec)

    meta_enc: MetadataEncoder | None = None
    if use_metadata:
        meta_enc = MetadataEncoder().fit(X_train)
    X_tr = combine_features(X_text_tr, meta_enc, X_train, use_metadata)
    X_va = combine_features(X_text_va, meta_enc, X_val, use_metadata)

    clf_s, clf_i = _fit_pair(X_tr, ys_tr, yi_tr, backend, random_state, calibration_cv)

    pred_s = clf_s.predict(X_va)
    pred_i = clf_i.predict(X_va)
    ps = clf_s.predict_proba(X_va)
    pi = clf_i.predict_proba(X_va)

    n_word = wv.get_feature_names_out().shape[0]
    n_char = cv_vec.get_feature_names_out().shape[0]

    return HoldoutResult(
        use_metadata=use_metadata,
        clf_state=clf_s,
        clf_intensity=clf_i,
        wv=wv,
        cv_vec=cv_vec,
        meta_enc=meta_enc,
        le_s=le_s,
        le_i=le_i,
        X_val=X_va,
        val_df=X_val.reset_index(drop=True),
        ys_va=ys_va,
        yi_va=yi_va,
        pred_state=pred_s,
        pred_intensity=pred_i,
        proba_state=ps,
        proba_intensity=pi,
        n_word=n_word,
        n_char=n_char,
    )
