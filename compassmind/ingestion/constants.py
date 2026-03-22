"""Shared column names and default artifact paths."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAINING_CSV = PACKAGE_ROOT / "Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv"
DEFAULT_TEST_PDF = PACKAGE_ROOT / "arvyax_test_inputs_120.xlsx - Sheet1.pdf"

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
