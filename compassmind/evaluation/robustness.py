"""Controlled robustness checks on held-out rows (no label leakage—descriptive shifts only)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from compassmind.features import MetadataEncoder, combine_features, transform_text_features
from sklearn.calibration import CalibratedClassifierCV


def _predict_row(
    df_one: pd.DataFrame,
    *,
    wv: Any,
    cv_vec: Any,
    meta_enc: MetadataEncoder | None,
    use_metadata: bool,
    clf: CalibratedClassifierCV,
    le: Any,
) -> str:
    Xt = transform_text_features(df_one, wv, cv_vec)
    X = combine_features(Xt, meta_enc, df_one, use_metadata)
    pred = clf.predict(X)[0]
    return le.inverse_transform([pred])[0]


def robustness_battery(
    base_df: pd.DataFrame,
    *,
    wv: Any,
    cv_vec: Any,
    meta_enc: MetadataEncoder | None,
    use_metadata: bool,
    clf_state: CalibratedClassifierCV,
    le_s: Any,
    sample_indices: list[int] | None = None,
    n_samples: int = 5,
) -> list[dict[str, Any]]:
    """
    Apply input perturbations; report how often predictions change vs baseline.

    Scenarios: ultra-short text, missing metadata, contradictory stress vs calm wording, heavy typos.
    """
    n = len(base_df)
    if sample_indices is None:
        rng = np.random.default_rng(42)
        sample_indices = sorted(rng.choice(n, size=min(n_samples, n), replace=False).tolist())

    results: list[dict[str, Any]] = []

    for idx in sample_indices:
        row = base_df.iloc[idx : idx + 1].copy()
        baseline = _predict_row(row, wv=wv, cv_vec=cv_vec, meta_enc=meta_enc, use_metadata=use_metadata, clf=clf_state, le=le_s)

        scenarios: dict[str, pd.DataFrame] = {
            "baseline": row,
            "text_ok": _perturb(row, journal="ok"),
            "text_fine": _perturb(row, journal="fine"),
            "text_missing_meta": _strip_metadata(row),
            "text_conflict_high_stress_calm_words": _perturb(
                row,
                journal="i feel calm and soft right now",
                stress=5.0,
                energy=2.0,
            ),
            "text_typo_heavy": _perturb(
                row,
                journal=(str(row["journal_text"].iloc[0]) + " strssed overwhelmmed cant foccus") * 1,
            ),
        }

        preds = {k: _predict_row(v, wv=wv, cv_vec=cv_vec, meta_enc=meta_enc, use_metadata=use_metadata, clf=clf_state, le=le_s) for k, v in scenarios.items()}

        results.append(
            {
                "row_index": idx,
                "baseline_state": baseline,
                "predictions_by_scenario": preds,
                "changed_vs_baseline": {k: preds[k] != baseline for k in preds if k != "baseline"},
            }
        )

    return results


def _perturb(
    row: pd.DataFrame,
    *,
    journal: str | None = None,
    stress: float | None = None,
    energy: float | None = None,
) -> pd.DataFrame:
    out = deepcopy(row)
    if journal is not None:
        out["journal_text"] = journal
    if stress is not None:
        out["stress_level"] = stress
    if energy is not None:
        out["energy_level"] = energy
    return out


def _strip_metadata(row: pd.DataFrame) -> pd.DataFrame:
    out = deepcopy(row)
    for c in [
        "ambience_type",
        "duration_min",
        "sleep_hours",
        "energy_level",
        "stress_level",
        "time_of_day",
        "previous_day_mood",
        "face_emotion_hint",
        "reflection_quality",
    ]:
        if c in out.columns:
            out[c] = np.nan
    return out
