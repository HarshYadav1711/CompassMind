"""Batch inference: probabilities, uncertainty, rule-based actions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from compassmind.decision import recommend
from compassmind.features import MetadataEncoder, combine_features, transform_text_features
from compassmind.uncertainty import combined_scores, uncertain_mask


def predict_dataframe(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    cfg = bundle["feature_config"]
    use_metadata = bundle["use_metadata"]
    wv, cv = bundle["wv"], bundle["cv"]
    meta_enc: MetadataEncoder | None = bundle.get("meta_enc")
    clf_s = bundle["clf_state"]
    clf_i = bundle["clf_intensity"]
    le_s = bundle["le_state"]
    le_i = bundle["le_intensity"]
    ct, et = bundle["conf_thresh"], bundle["ent_thresh"]

    X_text = transform_text_features(df, wv, cv)
    X = combine_features(X_text, meta_enc, df, use_metadata)

    ps = clf_s.predict_proba(X)
    pi = clf_i.predict_proba(X)
    ms, mi, conf, ent_s = combined_scores(ps, pi)
    um = uncertain_mask(ms, ent_s, ct, et)

    pred_state = le_s.inverse_transform(np.argmax(ps, axis=1))
    pred_int_raw = le_i.inverse_transform(np.argmax(pi, axis=1))
    pred_int = np.array([int(x) for x in pred_int_raw])

    rows = []
    for i in range(len(df)):
        r = df.iloc[i].to_dict()
        what, when = recommend(
            str(pred_state[i]),
            int(pred_int[i]),
            bool(um[i]),
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
                "uncertain_flag": bool(um[i]),
                "what_to_do": what,
                "when_to_do": when,
            }
        )
    return pd.DataFrame(rows)
