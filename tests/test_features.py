"""Feature matrix construction (metadata + missingness)."""

import pandas as pd

from compassmind.features import (
    FeatureConfig,
    MetadataEncoder,
    fit_transform_text_features,
    transform_text_features,
)


def test_metadata_encoder_includes_nine_missing_flags():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "journal_text": "hello world",
                "ambience_type": "ocean",
                "duration_min": 10.0,
                "sleep_hours": 7.0,
                "energy_level": 3.0,
                "stress_level": 2.0,
                "time_of_day": "morning",
                "previous_day_mood": None,
                "face_emotion_hint": "none",
                "reflection_quality": "clear",
            }
        ]
    )
    enc = MetadataEncoder().fit(df)
    X = enc.transform(df)
    # 4 scaled nums + 9 missing flags + one-hot (width depends on categories present)
    assert X.shape[0] == 1
    assert X.shape[1] >= 4 + 9 + 3


def test_text_features_stack_word_and_char():
    df = pd.DataFrame(
        [
            {"journal_text": "short noisy 4text", "id": 1},
            {"journal_text": "another line of noisy 4text", "id": 2},
        ]
    )
    cfg = FeatureConfig(max_word_features=50, max_char_features=40, min_df=1, max_df=1.0)
    X, wv, cv = fit_transform_text_features(df, cfg)
    X2 = transform_text_features(df, wv, cv)
    assert X.shape == X2.shape
