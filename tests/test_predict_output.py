"""Prediction DataFrame schema, shapes, and alignment with ``PredictionRow``."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from compassmind.features import FeatureConfig, combine_features, transform_text_features
from compassmind.predict import predict_dataframe
from compassmind.schemas import PredictionRow
from compassmind.train_eval import train_bundle

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = ROOT / "Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv"

EXPECTED_CSV_COLUMNS = [
    k for k in PredictionRow.model_fields.keys() if k != "supportive_message"
]


def test_expected_columns_match_predict_dataframe():
    assert EXPECTED_CSV_COLUMNS == [
        "id",
        "predicted_state",
        "predicted_intensity",
        "confidence",
        "uncertain_flag",
        "what_to_do",
        "when_to_do",
    ]


@pytest.mark.skipif(not CSV.is_file(), reason="training CSV not present")
def test_prediction_rows_validate_against_pydantic():
    df = pd.read_csv(CSV).head(120)
    cfg = FeatureConfig(max_word_features=400, max_char_features=300, random_state=0)
    bundle = train_bundle(df, cfg, use_metadata=True, random_state=0, run_stratified_cv=False)
    out = predict_dataframe(df, bundle)
    assert list(out.columns) == EXPECTED_CSV_COLUMNS
    for _, row in out.iterrows():
        PredictionRow.model_validate(row.to_dict())


@pytest.mark.skipif(not CSV.is_file(), reason="training CSV not present")
def test_predict_proba_shapes_match_encoders():
    df = pd.read_csv(CSV).head(120)
    cfg = FeatureConfig(max_word_features=400, max_char_features=300, random_state=0)
    bundle = train_bundle(df, cfg, use_metadata=True, random_state=0, run_stratified_cv=False)
    X_text = transform_text_features(df, bundle["wv"], bundle["cv"])
    X = combine_features(X_text, bundle.get("meta_enc"), df, bundle["use_metadata"])
    ps = bundle["clf_state"].predict_proba(X)
    pi = bundle["clf_intensity"].predict_proba(X)
    assert ps.shape == (len(df), len(bundle["le_state"].classes_))
    assert pi.shape == (len(df), len(bundle["le_intensity"].classes_))


@pytest.mark.skipif(not CSV.is_file(), reason="training CSV not present")
def test_prediction_row_count_matches_input():
    """Output length matches input rows (uses enough training rows for stratified split)."""
    # Enough rows so stratified holdout includes every emotional_state class (6+ in val).
    df = pd.read_csv(CSV).head(80)
    cfg = FeatureConfig(max_word_features=120, max_char_features=80, min_df=1, random_state=0)
    bundle = train_bundle(df, cfg, use_metadata=True, random_state=0, run_stratified_cv=False)
    out = predict_dataframe(df, bundle)
    assert len(out) == len(df)
