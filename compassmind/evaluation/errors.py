"""Extract validation failure cases for narrative error analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from compassmind.evaluation.holdout import HoldoutResult


def extract_failure_cases(
    hr: HoldoutResult,
    *,
    max_cases: int = 12,
    prefer_state_errors: bool = True,
) -> list[dict[str, Any]]:
    """
    Build structured failure rows for documentation.

    Prioritizes emotional_state mistakes, then intensity-only mistakes.
    """
    le_s, le_i = hr.le_s, hr.le_i
    rows: list[dict[str, Any]] = []
    state_err_idx: list[int] = []
    int_only_idx: list[int] = []

    for i in range(len(hr.val_df)):
        ts = le_s.inverse_transform([hr.ys_va[i]])[0]
        ps = le_s.inverse_transform([hr.pred_state[i]])[0]
        ti = int(le_i.inverse_transform([hr.yi_va[i]])[0])
        pi = int(le_i.inverse_transform([hr.pred_intensity[i]])[0])
        if ts != ps:
            state_err_idx.append(i)
        elif ti != pi:
            int_only_idx.append(i)

    order = state_err_idx + int_only_idx if prefer_state_errors else list(range(len(hr.val_df)))

    seen = 0
    for i in order:
        if seen >= max_cases:
            break
        ts = le_s.inverse_transform([hr.ys_va[i]])[0]
        ps = le_s.inverse_transform([hr.pred_state[i]])[0]
        ti = int(le_i.inverse_transform([hr.yi_va[i]])[0])
        pi = int(le_i.inverse_transform([hr.pred_intensity[i]])[0])
        if ts == ps and ti == pi:
            continue

        row = hr.val_df.iloc[i]
        journal = str(row.get("journal_text", ""))[:280]

        def _clean(v: object) -> object:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            if isinstance(v, (float, np.floating)):
                return round(float(v), 2)
            return v

        meta = {
            "stress": _clean(row.get("stress_level")),
            "energy": _clean(row.get("energy_level")),
            "time_of_day": row.get("time_of_day"),
            "face": row.get("face_emotion_hint"),
        }

        kind = "state" if ts != ps else "intensity_only"
        wrong, mechanism, fix = _explain(ts, ps, ti, pi, journal, meta, kind)

        rows.append(
            {
                "idx": int(i),
                "input_summary": {
                    "journal_excerpt": journal + ("…" if len(str(row.get("journal_text", ""))) > 280 else ""),
                    "metadata": meta,
                },
                "true_state": str(ts),
                "pred_state": str(ps),
                "true_intensity": ti,
                "pred_intensity": pi,
                "failure_kind": kind,
                "what_went_wrong": wrong,
                "why_it_failed": mechanism,
                "how_to_improve": fix,
            }
        )
        seen += 1

    return rows


def _explain(
    ts: str,
    ps: str,
    ti: int,
    pi: int,
    journal: str,
    meta: dict,
    kind: str,
) -> tuple[str, str, str]:
    """Prediction mistake (one line), mechanism, remediation."""
    n_words = len(journal.split())
    short = n_words < 10

    if kind == "state":
        wrong = f"Predicted `{ps}` instead of labeled `{ts}` (intensity {pi} vs {ti})."
    else:
        wrong = f"State correct (`{ts}`) but intensity {pi} vs labeled {ti}."

    mech: list[str] = []
    if short:
        mech.append("Bag-of-ngrams underfits when the journal has almost no tokens.")
    if kind == "state":
        mech.append(
            "Multiclass softmax must pick one label; overlapping vocabulary across states (e.g. tired vs overwhelmed) creates near ties."
        )
    else:
        mech.append(
            "Intensity is a 5-way classifier without ordinal structure—boundary errors near 3/4 are common when language is ambiguous."
        )
    st, en = meta.get("stress"), meta.get("energy")
    try:
        if st is not None and float(st) >= 4 and "calm" in journal.lower():
            mech.append("Calm wording coexists with high stress metadata; the model cannot represent 'both' in one class.")
    except (TypeError, ValueError):
        pass

    fixes = [
        "Add contrastive / hard-negative pairs for commonly confused state pairs.",
        "Try ordinal intensity (e.g. cumulative link) or cost-sensitive loss on adjacent bins.",
        "Use `uncertain_flag` + conservative UX whenever margin or entropy is poor (already in pipeline).",
    ]
    if short:
        fixes.insert(0, "Down-weight or abstain on ultra-short reflections in evaluation and product.")

    return wrong, " ".join(mech), " ".join(fixes[:3])
