"""Batch inference: probabilities, uncertainty layer, rule-based actions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from compassmind.decision import recommend
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
    le_i = bundle["le_intensity"]

    ucfg = build_uncertainty_config(bundle)

    X_text = transform_text_features(df, wv, cv)
    X = combine_features(X_text, meta_enc, df, use_metadata)

    ps = clf_s.predict_proba(X)
    pi = clf_i.predict_proba(X)
    ms, mi, conf, ent_s = combined_confidence(ps, pi)
    margin_state = top_two_margin(ps)

    pred_state = le_s.inverse_transform(np.argmax(ps, axis=1))
    pred_int_raw = le_i.inverse_transform(np.argmax(pi, axis=1))
    pred_int = np.array([int(x) for x in pred_int_raw])

    rows: list[dict[str, Any]] = []
    for i in range(len(df)):
        r = df.iloc[i].to_dict()
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
    return pd.DataFrame(rows)[cols]
