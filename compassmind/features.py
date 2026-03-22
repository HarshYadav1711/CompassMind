"""Feature pipelines: word + char TF-IDF and metadata with missing indicators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from compassmind.text import journal_for_features

CAT_COLS = [
    "ambience_type",
    "time_of_day",
    "previous_day_mood",
    "face_emotion_hint",
    "reflection_quality",
]
NUM_COLS = ["duration_min", "sleep_hours", "energy_level", "stress_level"]
NUM_MISSING_FLAGS = [f"{c}_is_missing" for c in NUM_COLS]


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in NUM_COLS:
        out[f"{c}_is_missing"] = out[c].isna().astype(np.float32)
    return out


def _fill_cat(s: pd.Series) -> pd.Series:
    return s.fillna("__MISSING__").astype(str)


@dataclass
class FeatureConfig:
    use_metadata: bool = True
    word_ngram_max: int = 2
    char_ngram_range: Tuple[int, int] = (3, 5)
    max_word_features: int = 12000
    max_char_features: int = 8000


def build_text_vectorizers(cfg: FeatureConfig) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    word_vec = TfidfVectorizer(
        ngram_range=(1, cfg.word_ngram_max),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=cfg.max_word_features,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.char_ngram_range,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=cfg.max_char_features,
    )
    return word_vec, char_vec


def fit_transform_text_features(
    df: pd.DataFrame, cfg: FeatureConfig
) -> Tuple[csr_matrix, TfidfVectorizer, TfidfVectorizer]:
    texts = df["journal_text"].fillna("").map(journal_for_features)
    wv, cv = build_text_vectorizers(cfg)
    Xw = wv.fit_transform(texts)
    Xc = cv.fit_transform(texts)
    return hstack([Xw, Xc], format="csr"), wv, cv


def transform_text_features(df: pd.DataFrame, wv: TfidfVectorizer, cv: TfidfVectorizer) -> csr_matrix:
    texts = df["journal_text"].fillna("").map(journal_for_features)
    return hstack([wv.transform(texts), cv.transform(texts)], format="csr")


class MetadataEncoder:
    """Numeric median impute + scale + missing flags; categorical OHE with missing bucket."""

    def __init__(self) -> None:
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        self._ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    def fit(self, df: pd.DataFrame) -> "MetadataEncoder":
        df2 = add_missing_indicators(df)
        num = df2[NUM_COLS].astype(float)
        self._imputer.fit(num)
        num_i = self._imputer.transform(num)
        self._scaler.fit(num_i)
        self._ohe.fit(df2[CAT_COLS].apply(_fill_cat))
        return self

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        df2 = add_missing_indicators(df)
        num = df2[NUM_COLS].astype(float)
        num_i = self._imputer.transform(num)
        num_s = self._scaler.transform(num_i)
        miss = df2[NUM_MISSING_FLAGS].astype(float).values
        dense = np.hstack([num_s, miss])
        cat_m = self._ohe.transform(df2[CAT_COLS].apply(_fill_cat))
        return hstack([sp.csr_matrix(dense), cat_m], format="csr")


def combine_features(
    X_text: csr_matrix, meta_enc: Optional[MetadataEncoder], df: pd.DataFrame, use_metadata: bool
) -> csr_matrix:
    if not use_metadata or meta_enc is None:
        return X_text
    X_meta = meta_enc.transform(df)
    return hstack([X_text, X_meta], format="csr")
