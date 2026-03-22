"""Load and validate the training CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from compassmind.ingestion.constants import (
    DEFAULT_TRAINING_CSV,
    FEATURE_COLUMN_ORDER,
    TRAINING_LABEL_COLUMNS,
)
from compassmind.ingestion.preprocess import preprocess_training_dataframe
from compassmind.ingestion.schema import validate_training_dataframe


def load_training_csv(
    path: str | Path | None = None,
    *,
    validate: bool = True,
    add_missing_flags: bool = False,
) -> pd.DataFrame:
    """
    Load training data from the Arvyax reflective dataset CSV.

    - Enforces expected columns
    - Applies deterministic preprocessing (see `preprocess_training_dataframe`)
    - Optionally validates each row against `TrainingRowStrict`
    - Optionally appends missingness flags (for downstream modeling)
    """
    csv_path = Path(path or DEFAULT_TRAINING_CSV)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    expected = list(FEATURE_COLUMN_ORDER) + list(TRAINING_LABEL_COLUMNS)
    df = pd.read_csv(csv_path)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV missing required columns {missing}. Found: {list(df.columns)}")

    df = df.loc[:, expected].copy()
    out = preprocess_training_dataframe(df)
    if validate:
        validate_training_dataframe(out)
    if add_missing_flags:
        from compassmind.ingestion.preprocess import add_missingness_flags

        out = add_missingness_flags(out)
        feat = list(FEATURE_COLUMN_ORDER) + list(TRAINING_LABEL_COLUMNS)
        rest = [c for c in out.columns if c not in feat]
        out = out.loc[:, feat + rest]
    return out
