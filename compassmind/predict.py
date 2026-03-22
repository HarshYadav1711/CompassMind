"""Batch inference: probabilities, uncertainty layer, rule-based actions."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from compassmind.decision import VALID_TIMINGS, recommend
from compassmind.features import MetadataEncoder, combine_features, transform_text_features
from compassmind.uncertainty import (
    build_uncertainty_config,
    combined_confidence,
    compute_uncertain_flag,
    top_two_margin,
    _conflicting_signals,
    _count_missing_metadata,
    _journal_weakness,
)


def _class_values_for_intensity_proba(clf_i: Any) -> np.ndarray:
    """Ordinal labels 1–5 aligned with ``predict_proba`` columns."""
    return np.array([int(float(c)) for c in clf_i.classes_], dtype=float)


def _expected_intensity_from_proba(pi: np.ndarray, class_values: np.ndarray) -> np.ndarray:
    """Soft expectation + round + clip (reduces single-class collapse vs argmax)."""
    ev = (pi * class_values).sum(axis=1)
    return np.clip(np.round(ev).astype(int), 1, 5)


def _float_field(row: dict[str, Any], key: str) -> float:
    """NaN if missing or non-numeric (do not impute defaults for signal rules)."""
    v = row.get(key)
    if v is None:
        return float("nan")
    if isinstance(v, float) and np.isnan(v):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _adjust_intensity_signals(pred: int, stress: float, energy: float, confidence: float) -> int:
    """Deterministic nudges from stress, energy, and low confidence (toward mid 3)."""
    p = int(np.clip(pred, 1, 5))
    if not np.isnan(stress) and stress >= 4.0 and p <= 3:
        p = min(5, p + 1)
    if not np.isnan(energy) and energy <= 2.0 and p >= 3:
        p = max(1, p - 1)
    if confidence < 0.45:
        if p > 3:
            p -= 1
        elif p < 3:
            p += 1
    return int(np.clip(p, 1, 5))


def validate_outputs(df: pd.DataFrame) -> None:
    """Raise if any ``when_to_do`` value is outside ``VALID_TIMINGS``."""
    if "when_to_do" not in df.columns:
        raise KeyError("predictions must include column 'when_to_do'")
    bad = df.loc[~df["when_to_do"].isin(VALID_TIMINGS), "when_to_do"]
    if len(bad) > 0:
        uniq = sorted(bad.unique().tolist(), key=str)
        raise ValueError(
            f"Invalid when_to_do value(s) {uniq}; allowed: {sorted(VALID_TIMINGS)}"
        )


def predict_dataframe(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    cfg = bundle["feature_config"]
    use_metadata = bundle["use_metadata"]
    wv = bundle["wv"]
    cv = bundle.get("cv", bundle.get("cv_vec"))
    if cv is None:
        raise KeyError("Bundle missing char TF-IDF vectorizer key 'cv'")
    meta_enc: MetadataEncoder | None = bundle.get("meta_enc")
    clf_s = bundle["clf_state"]
    clf_i = bundle["clf_intensity"]
    le_s = bundle["le_state"]

    ucfg = build_uncertainty_config(bundle)

    X_text = transform_text_features(df, wv, cv)
    X = combine_features(X_text, meta_enc, df, use_metadata)

    ps = clf_s.predict_proba(X)
    pi = clf_i.predict_proba(X)
    ms, mi, conf, ent_s = combined_confidence(ps, pi)
    margin_state = top_two_margin(ps)

    pred_state = le_s.inverse_transform(np.argmax(ps, axis=1))
    cls_vals = _class_values_for_intensity_proba(clf_i)
    pred_int = _expected_intensity_from_proba(pi, cls_vals)

    rows: list[dict[str, Any]] = []
    for i in range(len(df)):
        r = df.iloc[i].to_dict()
        st = _float_field(r, "stress_level")
        en = _float_field(r, "energy_level")
        pred_int[i] = _adjust_intensity_signals(int(pred_int[i]), st, en, float(conf[i]))

        jt = str(r.get("journal_text") or "")
        weak, _, _ = _journal_weakness(jt)
        miss_n = _count_missing_metadata(r)
        conflict = _conflicting_signals(r, str(pred_state[i]), int(pred_int[i]))

        uflag = compute_uncertain_flag(
            confidence=float(conf[i]),
            max_state_prob=float(ms[i]),
            norm_entropy_state=float(ent_s[i]),
            margin_state=float(margin_state[i]),
            journal_weak=weak,
            missing_meta=miss_n,
            conflicting=conflict,
            cfg=ucfg,
        )

        what, when = recommend(
            str(pred_state[i]),
            int(pred_int[i]),
            uflag,
            float(conf[i]),
            r.get("time_of_day"),
            r,
        )
        rows.append(
            {
                "id": int(r["id"]),
                "predicted_state": str(pred_state[i]),
                "predicted_intensity": int(pred_int[i]),
                "confidence": float(conf[i]),
                "uncertain_flag": int(uflag),
                "what_to_do": what,
                "when_to_do": when,
            }
        )
    cols = [
        "id",
        "predicted_state",
        "predicted_intensity",
        "confidence",
        "uncertain_flag",
        "what_to_do",
        "when_to_do",
    ]
    out = pd.DataFrame(rows)[cols]
    validate_outputs(out)
    return out


def main() -> None:
    """Support ``python -m compassmind.predict`` (same as ``python -m compassmind predict``)."""
    from compassmind.cli import main as cli_main

    cli_main(["predict", *sys.argv[1:]])


if __name__ == "__main__":
    main()
