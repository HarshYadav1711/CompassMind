"""Deterministic preprocessing: whitespace, lowercase journal text, missing standardization, flags."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Optional

import numpy as np
import pandas as pd

from compassmind.ingestion.constants import FEATURE_COLUMN_ORDER, MISSINGNESS_FLAG_COLUMNS, TRAINING_LABEL_COLUMNS

_WS_RE = re.compile(r"\s+")

# Strings treated as missing for optional fields (after strip).
_MISSING_STRINGS = frozenset(
    {
        "",
        "nan",
        "none",
        "null",
        "n/a",
        "na",
        "#n/a",
        "-",
        "--",
    }
)


def standardize_missing_scalar(value: Any) -> Any:
    """Map pandas/CSV sentinels and noisy strings to None for optional fields."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or np.isnan(value)):
        return None
    if isinstance(value, str):
        t = value.strip()
        if t.lower() in _MISSING_STRINGS:
            return None
        return t
    if pd.isna(value):
        return None
    return value


def _is_missing_value(value: Any) -> bool:
    v = standardize_missing_scalar(value)
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def preprocess_journal_text(text: str | None) -> str:
    """
    Normalize Unicode, collapse whitespace, trim, lowercase.

    Does not remove digits or fix spelling — typos and noisy tokens remain.
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = _WS_RE.sub(" ", t).strip().lower()
    return t


def _normalize_optional_categorical(raw: Any) -> Optional[str]:
    v = standardize_missing_scalar(raw)
    if v is None:
        return None
    if not isinstance(v, str):
        v = str(v)
    s = v.strip().lower()
    if s == "" or s in _MISSING_STRINGS:
        return None
    return s


def _normalize_optional_float(raw: Any) -> Optional[float]:
    v = standardize_missing_scalar(raw)
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if isinstance(v, float) and (math.isnan(v) or np.isnan(v)):
            return None
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in _MISSING_STRINGS:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _normalize_id(raw: Any) -> int:
    v = standardize_missing_scalar(raw)
    if v is None:
        raise ValueError("id is required")
    if isinstance(v, bool):
        raise ValueError("id must be an integer")
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError("id must be an integral value")
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s.isdigit():
            raise ValueError(f"id must be digits-only, got {raw!r}")
        return int(s)
    raise ValueError(f"Unsupported id type: {type(raw)}")


def standardize_feature_row_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Apply missing standardization and typing for one feature row (before strict schema)."""
    out: dict[str, Any] = {
        "id": _normalize_id(row.get("id")),
        "journal_text": preprocess_journal_text(row.get("journal_text")),
        "ambience_type": _normalize_optional_categorical(row.get("ambience_type")),
        "duration_min": _normalize_optional_float(row.get("duration_min")),
        "sleep_hours": _normalize_optional_float(row.get("sleep_hours")),
        "energy_level": _normalize_optional_float(row.get("energy_level")),
        "stress_level": _normalize_optional_float(row.get("stress_level")),
        "time_of_day": _normalize_optional_categorical(row.get("time_of_day")),
        "previous_day_mood": _normalize_optional_categorical(row.get("previous_day_mood")),
        "face_emotion_hint": _normalize_optional_categorical(row.get("face_emotion_hint")),
        "reflection_quality": _normalize_optional_categorical(row.get("reflection_quality")),
    }
    return out


def add_missingness_flags(df: pd.DataFrame, columns: tuple[str, ...] = MISSINGNESS_FLAG_COLUMNS) -> pd.DataFrame:
    """Append `{col}_is_missing` boolean columns (True if value is missing after standardization)."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[f"{col}_is_missing"] = True
            continue
        out[f"{col}_is_missing"] = out[col].map(lambda v: _is_missing_value(v))
    return out


def preprocess_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-wise standardization for feature tables.

    - journal_text: preprocess_journal_text
    - optional categoricals: strip, lower, empty -> None
    - optional numerics: parse float; invalid/empty -> None
    - id: coerce to int
    """
    rows: list[dict[str, Any]] = []
    for _, ser in df.iterrows():
        d = ser.to_dict()
        cleaned = standardize_feature_row_dict(d)
        rows.append(cleaned)
    return pd.DataFrame(rows, columns=list(FEATURE_COLUMN_ORDER))


def preprocess_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Like preprocess_feature_dataframe plus label columns coerced and stripped."""
    base = preprocess_feature_dataframe(df)
    if "emotional_state" not in df.columns or "intensity" not in df.columns:
        raise ValueError("Training frame must include emotional_state and intensity")
    states: list[str] = []
    intensities: list[int] = []
    for _, ser in df.iterrows():
        st = standardize_missing_scalar(ser.get("emotional_state"))
        if st is None or (isinstance(st, str) and st.strip() == ""):
            raise ValueError("emotional_state cannot be missing for training rows")
        states.append(str(st).strip().lower())
        raw_i = ser.get("intensity")
        iv = _normalize_optional_float(raw_i)
        if iv is None:
            raise ValueError("intensity cannot be missing for training rows")
        if abs(iv - round(iv)) > 1e-9:
            raise ValueError(f"intensity must be integral 1–5, got {raw_i!r}")
        intensities.append(int(round(iv)))
    out = base.copy()
    out["emotional_state"] = states
    out["intensity"] = intensities
    ordered = list(FEATURE_COLUMN_ORDER) + list(TRAINING_LABEL_COLUMNS)
    return out.loc[:, ordered]
