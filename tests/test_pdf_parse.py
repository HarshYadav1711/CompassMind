import pathlib

from compassmind.pdf_parse import parse_pdf_rows

ROOT = pathlib.Path(__file__).resolve().parents[1]
PDF = ROOT / "arvyax_test_inputs_120.xlsx - Sheet1.pdf"


def test_pdf_parses_120_rows():
    df = parse_pdf_rows(str(PDF))
    assert len(df) == 120
    assert df["id"].min() >= 10001
    assert df["journal_text"].str.len().gt(0).mean() > 0.5


def test_columns_present():
    df = parse_pdf_rows(str(PDF))
    for c in [
        "journal_text",
        "ambience_type",
        "duration_min",
        "sleep_hours",
        "energy_level",
        "stress_level",
        "time_of_day",
        "previous_day_mood",
        "face_emotion_hint",
        "reflection_quality",
    ]:
        assert c in df.columns
