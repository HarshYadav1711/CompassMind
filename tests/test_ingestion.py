"""Tests for CSV/PDF ingestion, schema validation, and preprocessing."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
from pydantic import ValidationError

from compassmind.ingestion.constants import DEFAULT_TEST_PDF, DEFAULT_TRAINING_CSV, FEATURE_COLUMN_ORDER
from compassmind.ingestion.csv_io import load_training_csv
from compassmind.ingestion.pdf_io import parse_test_pdf
from compassmind.ingestion.preprocess import (
    add_missingness_flags,
    preprocess_journal_text,
    standardize_missing_scalar,
)
from compassmind.ingestion.schema import (
    FeatureRowStrict,
    TrainingRowStrict,
    validate_features_dataframe,
    validate_training_dataframe,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not DEFAULT_TRAINING_CSV.is_file(), reason="training CSV not present")
def test_training_csv_loads_and_validates():
    df = load_training_csv(DEFAULT_TRAINING_CSV, validate=True)
    assert len(df) > 0
    assert list(df.columns[: len(FEATURE_COLUMN_ORDER)]) == list(FEATURE_COLUMN_ORDER)
    assert "emotional_state" in df.columns and "intensity" in df.columns


@pytest.mark.skipif(not DEFAULT_TEST_PDF.is_file(), reason="test PDF not present")
def test_pdf_reconstructs_tabular_features():
    df = parse_test_pdf(DEFAULT_TEST_PDF, validate=True, preprocess=True)
    assert len(df) == 120
    assert list(df.columns[:11]) == list(FEATURE_COLUMN_ORDER)
    assert df["id"].between(10001, 19999).all() or df["id"].min() >= 10001


def test_schema_rejects_invalid_feature_row():
    with pytest.raises(ValidationError):
        FeatureRowStrict.model_validate(
            {
                "id": -1,
                "journal_text": "x",
                "ambience_type": None,
                "duration_min": None,
                "sleep_hours": None,
                "energy_level": None,
                "stress_level": None,
                "time_of_day": None,
                "previous_day_mood": None,
                "face_emotion_hint": None,
                "reflection_quality": None,
            }
        )


def test_validate_features_dataframe_raises_on_bad_energy():
    bad = pd.DataFrame(
        [
            {
                "id": 1,
                "journal_text": "ok",
                "ambience_type": None,
                "duration_min": None,
                "sleep_hours": None,
                "energy_level": 99.0,
                "stress_level": None,
                "time_of_day": None,
                "previous_day_mood": None,
                "face_emotion_hint": None,
                "reflection_quality": None,
            }
        ]
    )
    with pytest.raises(ValueError, match="Feature schema validation failed"):
        validate_features_dataframe(bad)


def test_preprocess_preserves_noisy_tokens():
    raw = "  Organiz4ed  menta8l.L5y. idk  "
    out = preprocess_journal_text(raw)
    assert "organiz4ed" in out
    assert "menta8l.l5y." in out
    assert "idk" in out
    assert out == out.lower()
    assert not out.startswith(" ")


def test_standardize_missing_variants():
    assert standardize_missing_scalar("  NaN ") is None
    assert standardize_missing_scalar("N/A") is None
    assert standardize_missing_scalar(None) is None


def test_missingness_flags_are_deterministic():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "journal_text": "x",
                "ambience_type": None,
                "duration_min": None,
                "sleep_hours": 7.0,
                "energy_level": 3.0,
                "stress_level": None,
                "time_of_day": "morning",
                "previous_day_mood": None,
                "face_emotion_hint": "none",
                "reflection_quality": "clear",
            }
        ]
    )
    out = add_missingness_flags(df)
    assert bool(out["duration_min_is_missing"].iloc[0])
    assert not bool(out["sleep_hours_is_missing"].iloc[0])


def test_validate_training_dataframe_rejects_bad_state():
    bad = pd.DataFrame(
        [
            {
                "id": 1,
                "journal_text": "ok",
                "ambience_type": None,
                "duration_min": None,
                "sleep_hours": None,
                "energy_level": 3.0,
                "stress_level": 3.0,
                "time_of_day": "morning",
                "previous_day_mood": "neutral",
                "face_emotion_hint": "none",
                "reflection_quality": "clear",
                "emotional_state": "not_a_real_label",
                "intensity": 3,
            }
        ]
    )
    with pytest.raises(ValueError, match="Training schema validation failed"):
        validate_training_dataframe(bad)


def test_training_row_strict_accepts_valid_labels():
    TrainingRowStrict.model_validate(
        {
            "id": 1,
            "journal_text": "hello",
            "ambience_type": "ocean",
            "duration_min": 10.0,
            "sleep_hours": 7.5,
            "energy_level": 3.0,
            "stress_level": 2.0,
            "time_of_day": "morning",
            "previous_day_mood": "calm",
            "face_emotion_hint": "calm_face",
            "reflection_quality": "clear",
            "emotional_state": "calm",
            "intensity": 3,
        }
    )
