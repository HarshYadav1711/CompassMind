import pathlib

import pandas as pd

from compassmind.features import FeatureConfig
from compassmind.train_eval import train_bundle

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = ROOT / "Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv"


def test_train_bundle_runs_small_subsample():
    df = pd.read_csv(CSV).head(200)
    cfg = FeatureConfig(max_word_features=500, max_char_features=400)
    bundle = train_bundle(df, cfg, use_metadata=True, random_state=0, run_stratified_cv=False)
    assert "clf_state" in bundle and "metrics" in bundle
