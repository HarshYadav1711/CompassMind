"""Shared column names; dataset paths align with ``compassmind.constants``."""

from __future__ import annotations

from compassmind.constants import (
    DEFAULT_TEST_PDF,
    DEFAULT_TRAINING_CSV,
    PROJECT_ROOT as PACKAGE_ROOT,
)

# Input features shared by training (minus labels) and test.
FEATURE_COLUMN_ORDER: tuple[str, ...] = (
    "id",
    "journal_text",
    "ambience_type",
    "duration_min",
    "sleep_hours",
    "energy_level",
    "stress_level",
    "time_of_day",
    "previous_day_mood",
    "face_emotion_hint",
    "reflection_quality",
)

TRAINING_LABEL_COLUMNS: tuple[str, ...] = ("emotional_state", "intensity")

# Metadata fields that get explicit missingness flags after standardization.
MISSINGNESS_FLAG_COLUMNS: tuple[str, ...] = (
    "ambience_type",
    "duration_min",
    "sleep_hours",
    "energy_level",
    "stress_level",
    "time_of_day",
    "previous_day_mood",
    "face_emotion_hint",
    "reflection_quality",
)
