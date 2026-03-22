"""Strict Pydantic schemas for tabular ingestion."""

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from compassmind.ingestion.constants import FEATURE_COLUMN_ORDER, TRAINING_LABEL_COLUMNS

TimeOfDay = Literal["morning", "afternoon", "evening", "night", "early_morning"]
AmbienceType = Literal["ocean", "forest", "mountain", "rain", "cafe"]
FaceHint = Literal["calm_face", "tired_face", "tense_face", "happy_face", "neutral_face", "none"]
ReflectionQuality = Literal["clear", "vague", "conflicted"]
PreviousMood = Literal["calm", "mixed", "neutral", "overwhelmed", "restless", "focused"]
EmotionalState = Literal["calm", "focused", "mixed", "neutral", "overwhelmed", "restless"]


class FeatureRowStrict(BaseModel):
    """One row of model inputs (training without labels, or test PDF)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: int = Field(gt=0, description="Positive integer row id")
    journal_text: str = Field(min_length=0)
    ambience_type: Optional[AmbienceType] = None
    duration_min: Optional[float] = Field(default=None, ge=0.0, le=120.0)
    sleep_hours: Optional[float] = Field(default=None, ge=0.0, le=24.0)
    energy_level: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    stress_level: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    time_of_day: Optional[TimeOfDay] = None
    previous_day_mood: Optional[PreviousMood] = None
    face_emotion_hint: Optional[FaceHint] = None
    reflection_quality: Optional[ReflectionQuality] = None


class TrainingRowStrict(FeatureRowStrict):
    """Training row including labels."""

    emotional_state: EmotionalState
    intensity: int = Field(ge=1, le=5)


def _series_to_dict(row: pd.Series) -> dict:
    """Convert a pandas row to a plain dict with None instead of NaN."""
    d: dict = {}
    for k, v in row.items():
        if pd.isna(v):
            d[k] = None
        else:
            d[k] = v
    return d


def _subset_dict(row: pd.Series, keys: tuple[str, ...]) -> dict:
    d = _series_to_dict(row)
    return {k: d.get(k) for k in keys}


def validate_features_dataframe(df: pd.DataFrame) -> None:
    """Validate every row; raises ValueError with details on first failure."""
    keys = FEATURE_COLUMN_ORDER
    for pos, (_, row) in enumerate(df.iterrows()):
        try:
            FeatureRowStrict.model_validate(_subset_dict(row, keys))
        except ValidationError as e:
            raise ValueError(f"Feature schema validation failed at dataframe index {pos} (id={row.get('id')}): {e}") from e


def validate_training_dataframe(df: pd.DataFrame) -> None:
    """Validate every training row including labels."""
    keys = tuple(list(FEATURE_COLUMN_ORDER) + list(TRAINING_LABEL_COLUMNS))
    for pos, (_, row) in enumerate(df.iterrows()):
        try:
            TrainingRowStrict.model_validate(_subset_dict(row, keys))
        except ValidationError as e:
            raise ValueError(
                f"Training schema validation failed at dataframe index {pos} (id={row.get('id')}): {e}"
            ) from e
