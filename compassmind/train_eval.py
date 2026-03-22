"""Baseline training: calibrated linear models, optional XGBoost benchmark, ablation, artifacts."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from compassmind.features import (
    FeatureConfig,
    MetadataEncoder,
    combine_features,
    fit_transform_text_features,
    transform_text_features,
)
from compassmind.uncertainty import combined_scores, uncertain_mask

Backend = Literal["logistic", "xgboost"]

BASE_DIR = Path(__file__).resolve().parents[1]


def prepare_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, LabelEncoder, LabelEncoder]:
    le_s = LabelEncoder()
    le_i = LabelEncoder()
    y_s = le_s.fit_transform(df["emotional_state"].astype(str))
    y_i = le_i.fit_transform(df["intensity"].astype(int).astype(str))
    return y_s, y_i, le_s, le_i


def _build_logistic_base(random_state: int, *, max_iter: int = 5000, C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(
        solver="saga",
        penalty="l2",
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def build_calibrated_state_clf(
    backend: Backend,
    random_state: int,
    calibration_cv: int = 3,
) -> CalibratedClassifierCV:
    if backend == "logistic":
        base = _build_logistic_base(random_state, max_iter=5000, C=1.0)
    else:
        base = _build_xgb_multiclass(random_state)
    return CalibratedClassifierCV(base, method="sigmoid", cv=calibration_cv, n_jobs=-1)


def build_calibrated_intensity_clf(
    backend: Backend,
    random_state: int,
    calibration_cv: int = 3,
) -> CalibratedClassifierCV:
    if backend == "logistic":
        base = _build_logistic_base(random_state + 1, max_iter=6000, C=1.5)
    else:
        base = _build_xgb_multiclass(random_state + 1)
    return CalibratedClassifierCV(base, method="sigmoid", cv=calibration_cv, n_jobs=-1)


def _build_xgb_multiclass(random_state: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install xgboost for backend='xgboost'") from e
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1.0,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
        eval_metric="mlogloss",
    )


def _tune_uncertainty_thresholds(
    clf_s: CalibratedClassifierCV,
    X_va: Any,
    ys_va: np.ndarray,
    yi_va: np.ndarray,
    clf_i: CalibratedClassifierCV,
) -> tuple[float, float, dict[str, float]]:
    ps_va = clf_s.predict_proba(X_va)
    pi_va = clf_i.predict_proba(X_va)
    ms, mi, conf, ent_s = combined_scores(ps_va, pi_va)

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

    aux = {
        "val_intensity_accuracy": float(accuracy_score(yi_va, clf_i.predict(X_va))),
        "val_intensity_macro_f1": float(f1_score(yi_va, clf_i.predict(X_va), average="macro")),
    }
    return conf_thresh, ent_thresh, aux


def stratified_cv_scores_state(
    X: Any,
    y: np.ndarray,
    *,
    backend: Backend,
    random_state: int,
    n_splits: int = 5,
    calibration_cv: int = 3,
) -> dict[str, float]:
    """Stratified K-fold CV for emotional_state only (macro-F1, accuracy, log-loss)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    f1s: list[float] = []
    accs: list[float] = []
    lls: list[float] = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        clf = build_calibrated_state_clf(backend, random_state + fold, calibration_cv=calibration_cv)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[va])
        proba = clf.predict_proba(X[va])
        f1s.append(f1_score(y[va], pred, average="macro"))
        accs.append(accuracy_score(y[va], pred))
        try:
            lls.append(log_loss(y[va], proba))
        except ValueError:
            lls.append(float("nan"))
    return {
        f"cv{n_splits}_state_macro_f1_mean": float(np.nanmean(f1s)),
        f"cv{n_splits}_state_macro_f1_std": float(np.nanstd(f1s)),
        f"cv{n_splits}_state_accuracy_mean": float(np.nanmean(accs)),
        f"cv{n_splits}_state_logloss_mean": float(np.nanmean(lls)),
    }


def _fit_models_on_train(
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


def _evaluate_val(
    clf_s: CalibratedClassifierCV,
    clf_i: CalibratedClassifierCV,
    X_va: Any,
    ys_va: np.ndarray,
    yi_va: np.ndarray,
) -> dict[str, float]:
    ps_va = clf_s.predict_proba(X_va)
    pred_s = clf_s.predict(X_va)
    metrics: dict[str, float] = {
        "val_state_accuracy": float(accuracy_score(ys_va, pred_s)),
        "val_state_macro_f1": float(f1_score(ys_va, pred_s, average="macro")),
        "val_state_logloss": float(log_loss(ys_va, ps_va)),
    }
    metrics["val_intensity_accuracy"] = float(accuracy_score(yi_va, clf_i.predict(X_va)))
    metrics["val_intensity_macro_f1"] = float(f1_score(yi_va, clf_i.predict(X_va), average="macro"))
    return metrics


def benchmark_xgboost_vs_logistic(
    X_tr: Any,
    X_va: Any,
    ys_tr: np.ndarray,
    ys_va: np.ndarray,
    yi_tr: np.ndarray,
    yi_va: np.ndarray,
    random_state: int,
    calibration_cv: int = 3,
    min_macro_f1_gain: float = 0.012,
) -> tuple[Backend, dict[str, Any]]:
    """
    Train logistic and XGBoost (if available); pick XGBoost only if state macro-F1 improves meaningfully.
    """
    log_s, log_i = _fit_models_on_train(X_tr, ys_tr, yi_tr, "logistic", random_state, calibration_cv)
    log_metrics = _evaluate_val(log_s, log_i, X_va, ys_va, yi_va)
    out: dict[str, Any] = {"logistic": log_metrics}

    try:
        xgb_s, xgb_i = _fit_models_on_train(X_tr, ys_tr, yi_tr, "xgboost", random_state, calibration_cv)
        xgb_metrics = _evaluate_val(xgb_s, xgb_i, X_va, ys_va, yi_va)
        out["xgboost"] = xgb_metrics
        gain = xgb_metrics["val_state_macro_f1"] - log_metrics["val_state_macro_f1"]
        if gain >= min_macro_f1_gain:
            return "xgboost", {
                "chosen": "xgboost",
                "val_state_macro_f1_gain_vs_logistic": float(gain),
                "comparison": out,
            }
    except ImportError:
        return "logistic", {"chosen": "logistic", "xgboost": "not_installed", "comparison": out}
    except Exception as e:  # pragma: no cover
        warnings.warn(f"XGBoost benchmark skipped: {e}", UserWarning)
        return "logistic", {"chosen": "logistic", "xgboost_error": str(e), "comparison": out}

    return "logistic", {
        "chosen": "logistic",
        "val_state_macro_f1_gain_vs_logistic": float(
            out["xgboost"]["val_state_macro_f1"] - out["logistic"]["val_state_macro_f1"]
        ),
        "comparison": out,
    }


def train_bundle(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    use_metadata: bool,
    *,
    random_state: int = 42,
    test_size: float = 0.15,
    calibration_cv: int = 3,
    backend: Backend = "logistic",
    try_xgb_benchmark: bool = False,
    run_stratified_cv: bool = True,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """
    Fit calibrated classifiers, tune uncertainty thresholds on a stratified holdout,
    refit on full data, return artifact bundle + metrics.
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

    X_text_tr, wv, cv = fit_transform_text_features(X_train, cfg)
    X_text_va = transform_text_features(X_val, wv, cv)

    meta_enc: Optional[MetadataEncoder] = None
    if use_metadata:
        meta_enc = MetadataEncoder().fit(X_train)
    X_tr = combine_features(X_text_tr, meta_enc, X_train, use_metadata)
    X_va = combine_features(X_text_va, meta_enc, X_val, use_metadata)

    bench_info: dict[str, Any] = {}
    chosen_backend = backend
    if try_xgb_benchmark and backend == "logistic":
        chosen_backend, bench_info = benchmark_xgboost_vs_logistic(
            X_tr, X_va, ys_tr, ys_va, yi_tr, yi_va, random_state, calibration_cv=calibration_cv
        )

    clf_s, clf_i = _fit_models_on_train(X_tr, ys_tr, yi_tr, chosen_backend, random_state, calibration_cv)

    val_metrics = _evaluate_val(clf_s, clf_i, X_va, ys_va, yi_va)
    conf_thresh, ent_thresh, aux = _tune_uncertainty_thresholds(clf_s, X_va, ys_va, yi_va, clf_i)
    val_metrics.update(aux)

    ps_va = clf_s.predict_proba(X_va)
    ms, _, _, ent_s = combined_scores(ps_va, clf_i.predict_proba(X_va))
    val_metrics["val_uncertain_rate_tuned"] = float(uncertain_mask(ms, ent_s, conf_thresh, ent_thresh).mean())
    val_metrics["conf_thresh"] = conf_thresh
    val_metrics["ent_thresh"] = ent_thresh
    val_metrics["use_metadata"] = use_metadata
    val_metrics["backend"] = chosen_backend

    X_full_text, wv_f, cv_f = fit_transform_text_features(df, cfg)
    meta_full: Optional[MetadataEncoder] = None
    if use_metadata:
        meta_full = MetadataEncoder().fit(df)
    X_f = combine_features(X_full_text, meta_full, df, use_metadata)

    cv_metrics: dict[str, float] = {}
    if run_stratified_cv:
        cv_metrics = stratified_cv_scores_state(
            X_f,
            y_s,
            backend=chosen_backend,
            random_state=random_state,
            n_splits=cv_folds,
            calibration_cv=calibration_cv,
        )

    clf_s_f, clf_i_f = _fit_models_on_train(X_f, y_s, y_i, chosen_backend, random_state, calibration_cv)

    bundle: dict[str, Any] = {
        "feature_config": cfg,
        "use_metadata": use_metadata,
        "backend": chosen_backend,
        "random_state": random_state,
        "test_size": test_size,
        "calibration_cv": calibration_cv,
        "wv": wv_f,
        "cv": cv_f,
        "meta_enc": meta_full,
        "clf_state": clf_s_f,
        "clf_intensity": clf_i_f,
        "le_state": le_s,
        "le_intensity": le_i,
        "conf_thresh": conf_thresh,
        "ent_thresh": ent_thresh,
        "metrics": val_metrics,
        "cv_metrics": cv_metrics,
        "benchmark": bench_info,
        "label_classes_state": list(le_s.classes_),
        "label_classes_intensity": list(le_i.classes_),
    }
    return bundle


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Joblib may warn on large objects; acceptable for baseline artifacts.
    joblib.dump(bundle, path)
    payload = {
        "metrics": bundle.get("metrics"),
        "cv_metrics": bundle.get("cv_metrics"),
        "benchmark": bundle.get("benchmark"),
        "feature_config": asdict(bundle["feature_config"]),
        "backend": bundle.get("backend"),
        "use_metadata": bundle.get("use_metadata"),
        "random_state": bundle.get("random_state"),
    }
    with path.with_suffix(".metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_bundle(path: Path) -> dict[str, Any]:
    return joblib.load(path)
