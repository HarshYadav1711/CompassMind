"""High-level ingestion entry points for later stages (training, inference)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from compassmind.ingestion.constants import DEFAULT_TEST_PDF, DEFAULT_TRAINING_CSV
from compassmind.ingestion.csv_io import load_training_csv
from compassmind.ingestion.pdf_io import parse_test_pdf


def load_training_features(
    path: str | Path | None = None,
    *,
    validate: bool = True,
    add_missing_flags: bool = False,
) -> pd.DataFrame:
    """Load training CSV with preprocessing + strict validation."""
    return load_training_csv(path, validate=validate, add_missing_flags=add_missing_flags)


def load_test_pdf_features(
    path: str | Path | None = None,
    *,
    validate: bool = True,
    preprocess: bool = True,
    add_missing_flags: bool = False,
) -> pd.DataFrame:
    """Parse test PDF into the same feature schema as training (minus labels)."""
    return parse_test_pdf(
        path or DEFAULT_TEST_PDF,
        validate=validate,
        preprocess=preprocess,
        add_missing_flags=add_missing_flags,
    )
