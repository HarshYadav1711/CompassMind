"""Assignment-compliant ``when_to_do`` mapping and validation."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from compassmind.decision import VALID_TIMINGS, map_timing_label, recommend
from compassmind.predict import validate_outputs


def test_map_timing_label_passes_through_valid():
    for v in VALID_TIMINGS:
        assert map_timing_label(v, "morning") == v


def test_map_timing_label_internal_aliases():
    assert map_timing_label("after_break", "afternoon") == "within_15_min"
    assert map_timing_label("when_steady", "morning") == "within_15_min"
    assert map_timing_label("this_evening", "evening") == "tonight"
    assert map_timing_label("later", "morning") == "later_today"
    assert map_timing_label("soon", "night") == "within_15_min"


def test_map_timing_label_unknown_fallback():
    assert map_timing_label("totally_unknown", "night") == "tonight"
    assert map_timing_label("totally_unknown", "morning") == "later_today"


def test_recommend_only_emits_valid_timings():
    row = {"stress_level": 2.0, "energy_level": 3.0}
    _, when = recommend("focused", 3, 1, 0.2, "morning", row)
    assert when in VALID_TIMINGS


def test_validate_outputs_accepts_clean_frame():
    df = pd.DataFrame(
        {
            "when_to_do": list(VALID_TIMINGS) + ["now"],
        }
    )
    validate_outputs(df)


def test_validate_outputs_rejects_bad():
    df = pd.DataFrame({"when_to_do": ["after_break", "now"]})
    with pytest.raises(ValueError, match="Invalid when_to_do"):
        validate_outputs(df)


def test_mapping_logs_at_info(caplog):
    caplog.set_level(logging.INFO)
    map_timing_label("after_break", "afternoon")
    assert "Mapped timing" in caplog.text
