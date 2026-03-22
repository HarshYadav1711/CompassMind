"""Train calibrated classifiers and tune uncertainty thresholds on a validation split."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from compassmind.features import (
    FeatureConfig,
    MetadataEncoder,
    combine_features,
    fit_transform_text_features,
    transform_text_features,
)
from compassmind.uncertainty import combined_scores, uncertain_mask

BASE_DIR = Path(__file__).resolve().parents[1]


def prepare_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, LabelEncoder, LabelEncoder]:
    le_s = LabelEncoder()
    le_i = LabelEncoder()
    y_s = le_s.fit_transform(df["emotional_state"].astype(str))
    y_i = le_i.fit_transform(df["intensity"].astype(int).astype(str))
    return y_s, y_i, le_s, le_i


def build_state_clf() -> CalibratedClassifierCV:
    base = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=4000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=-1)


def build_intensity_clf() -> CalibratedClassifierCV:
    # Ordinal-ish 1–5 treated as 5-way classification; balanced for skew.
    base = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.5,
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=43,
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=-1)


def train_bundle(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    use_metadata: bool,
    random_state: int = 42,
) -> dict[str, Any]:
    y_s, y_i, le_s, le_i = prepare_labels(df)
    X_train, X_val, ys_tr, ys_va, yi_tr, yi_va = train_test_split(
        df,
        y_s,
        y_i,
        test_size=0.15,
        random_state=random_state,
        stratify=y_s,
    )

    X_text_tr, wv, cv = fit_transform_text_features(X_train, cfg)
    X_text_va = transform_text_features(X_val, wv, cv)

    meta_enc: Optional[MetadataEncoder] = None
    if use_metadata:
        meta_enc = MetadataEncoder().fit(X_train)
    X_tr = combine_features(X_text_tr, meta_enc, X_train, use_metadata)
    X_va = combine_features(X_text_va, meta_enc, X_val, use_metadata)

    clf_s = build_state_clf()
    clf_i = build_intensity_clf()
    clf_s.fit(X_tr, ys_tr)
    clf_i.fit(X_tr, yi_tr)

    ps_va = clf_s.predict_proba(X_va)
    pi_va = clf_i.predict_proba(X_va)
    ms, mi, conf, ent_s = combined_scores(ps_va, pi_va)

    # Tune thresholds to target ~14–30% uncertain on validation (state-based mask)
    best = (0.48, 0.86)
    best_score = -1.0
    for ct in np.linspace(0.35, 0.62, 15):
        for et in np.linspace(0.78, 0.95, 12):
            um = uncertain_mask(ms, ent_s, ct, et)
            rate = um.mean()
            if 0.12 <= rate <= 0.35:
                pred_s = clf_s.predict(X_va)
                f1 = f1_score(ys_va, pred_s, average="macro")
                score = f1 + 0.06 * (1.0 - abs(rate - 0.22))
                if score > best_score:
                    best_score = score
                    best = (float(ct), float(et))

    conf_thresh, ent_thresh = best
    if best_score < 0:
        # fallback if grid never hits band
        best_score = -1.0
        for ct in np.linspace(0.38, 0.58, 11):
            for et in np.linspace(0.82, 0.94, 7):
                um = uncertain_mask(ms, ent_s, ct, et)
                rate = um.mean()
                pred_s = clf_s.predict(X_va)
                f1 = f1_score(ys_va, pred_s, average="macro")
                score = f1 - 0.4 * abs(rate - 0.22)
                if score > best_score:
                    best_score = score
                    best = (float(ct), float(et))
        conf_thresh, ent_thresh = best

    metrics = {
        "val_state_accuracy": float(accuracy_score(ys_va, clf_s.predict(X_va))),
        "val_state_macro_f1": float(f1_score(ys_va, clf_s.predict(X_va), average="macro")),
        "val_intensity_accuracy": float(accuracy_score(yi_va, clf_i.predict(X_va))),
        "val_intensity_macro_f1": float(f1_score(yi_va, clf_i.predict(X_va), average="macro")),
        "val_state_logloss": float(log_loss(ys_va, ps_va)),
        "val_uncertain_rate_tuned": float(uncertain_mask(ms, ent_s, conf_thresh, ent_thresh).mean()),
        "conf_thresh": conf_thresh,
        "ent_thresh": ent_thresh,
        "use_metadata": use_metadata,
    }

    # Refit on full data for shipping artifact
    X_full, wv_f, cv_f = fit_transform_text_features(df, cfg)
    meta_full: Optional[MetadataEncoder] = None
    if use_metadata:
        meta_full = MetadataEncoder().fit(df)
    X_f = combine_features(X_full, meta_full, df, use_metadata)

    clf_s_f = build_state_clf()
    clf_i_f = build_intensity_clf()
    clf_s_f.fit(X_f, y_s)
    clf_i_f.fit(X_f, y_i)

    bundle = {
        "feature_config": cfg,
        "use_metadata": use_metadata,
        "wv": wv_f,
        "cv": cv_f,
        "meta_enc": meta_full,
        "clf_state": clf_s_f,
        "clf_intensity": clf_i_f,
        "le_state": le_s,
        "le_intensity": le_i,
        "conf_thresh": conf_thresh,
        "ent_thresh": ent_thresh,
        "metrics": metrics,
    }
    return bundle


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    with path.with_suffix(".metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"metrics": bundle["metrics"], "feature_config": asdict(bundle["feature_config"])},
            f,
            indent=2,
        )


def load_bundle(path: Path) -> dict[str, Any]:
    return joblib.load(path)
