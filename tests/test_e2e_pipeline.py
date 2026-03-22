"""Train → predict on shipped fixtures (skipped if files missing)."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from compassmind.cli import main
from compassmind.constants import DEFAULT_TEST_PDF, DEFAULT_TRAINING_CSV

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not DEFAULT_TRAINING_CSV.is_file(), reason="training CSV not present")
@pytest.mark.skipif(not DEFAULT_TEST_PDF.is_file(), reason="test PDF not present")
def test_cli_train_no_cv_then_predict_pdf(tmp_path):
    bundle = tmp_path / "bundle.joblib"
    pred = tmp_path / "out.csv"
    main(
        [
            "train",
            "--data",
            str(DEFAULT_TRAINING_CSV),
            "--artifacts",
            str(bundle),
            "--no-cv",
            "--seed",
            "42",
        ]
    )
    assert bundle.is_file()
    main(
        [
            "predict",
            "--bundle",
            str(bundle),
            "--pdf",
            str(DEFAULT_TEST_PDF),
            "--out",
            str(pred),
            "--seed",
            "42",
        ]
    )
    df = pd.read_csv(pred)
    assert len(df) == 120
    assert list(df.columns) == [
        "id",
        "predicted_state",
        "predicted_intensity",
        "confidence",
        "uncertain_flag",
        "what_to_do",
        "when_to_do",
    ]
