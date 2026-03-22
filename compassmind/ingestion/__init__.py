"""
Data ingestion: CSV training load, PDF test parse, strict schema validation, preprocessing.

This package is intentionally independent of model training so downstream stages can reuse it.
"""

from compassmind.ingestion.constants import (
    FEATURE_COLUMN_ORDER,
    MISSINGNESS_FLAG_COLUMNS,
    TRAINING_LABEL_COLUMNS,
)
from compassmind.ingestion.csv_io import load_training_csv
from compassmind.ingestion.pdf_io import parse_test_pdf
from compassmind.ingestion.pipeline import (
    load_test_pdf_features,
    load_training_features,
)
from compassmind.ingestion.preprocess import (
    add_missingness_flags,
    preprocess_feature_dataframe,
    standardize_missing_scalar,
)
from compassmind.ingestion.schema import (
    FeatureRowStrict,
    TrainingRowStrict,
    validate_features_dataframe,
    validate_training_dataframe,
)

__all__ = [
    "FEATURE_COLUMN_ORDER",
    "MISSINGNESS_FLAG_COLUMNS",
    "TRAINING_LABEL_COLUMNS",
    "load_training_csv",
    "parse_test_pdf",
    "load_training_features",
    "load_test_pdf_features",
    "add_missingness_flags",
    "preprocess_feature_dataframe",
    "standardize_missing_scalar",
    "FeatureRowStrict",
    "TrainingRowStrict",
    "validate_features_dataframe",
    "validate_training_dataframe",
]
