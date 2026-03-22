"""Parse the noisy test PDF into a feature dataframe (pdfplumber)."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pdfplumber

from compassmind.ingestion.constants import DEFAULT_TEST_PDF, FEATURE_COLUMN_ORDER
from compassmind.ingestion.preprocess import preprocess_feature_dataframe

KNOWN_TIME = frozenset({"morning", "afternoon", "evening", "night", "early_morning"})
KNOWN_MOOD = frozenset({"calm", "mixed", "neutral", "overwhelmed", "restless", "focused"})
KNOWN_AMBIENCE = frozenset({"ocean", "forest", "mountain", "rain", "cafe"})
FACE_PREFIXES = ("calm_face", "tired_face", "tense_face", "happy_face", "neutral_face", "none")
QUALITY = frozenset({"clear", "vague", "conflicted"})


def _parse_meta_float(token: str) -> Optional[float]:
    t = token.strip().strip(",.")
    t = re.sub(r"\.{2,}.*$", "", t)
    if not t:
        return None
    if not re.fullmatch(r"\d+(\.\d+)?", t):
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    if not (0 < v <= 120):
        return None
    return v


def _split_face_quality(blob: str) -> tuple[Optional[str], Optional[str]]:
    blob = blob.strip()
    if not blob:
        return None, None
    for face in FACE_PREFIXES:
        if blob.startswith(face):
            rest = blob[len(face) :]
            if rest in QUALITY:
                return face, rest
            if not rest:
                return face, None
    parts = blob.split()
    if len(parts) >= 2 and parts[-1] in QUALITY:
        face_guess = parts[-2]
        if face_guess.endswith("_face") or face_guess == "none":
            return face_guess, parts[-1]
    return None, None


def _parse_line_words(rid: int, words: list[dict[str, Any]]) -> dict[str, Any]:
    duration = sleep = energy = stress = None
    time_of_day = previous_mood = None
    face_hint = reflection_quality = None
    journal_parts: list[str] = []
    tail_blob = ""

    for w in words:
        x = w["x0"]
        t = w["text"]
        if x < 200:
            journal_parts.append(t)
            continue
        if 200 <= x < 245:
            v = _parse_meta_float(t)
            if v is not None:
                duration = v
            else:
                journal_parts.append(t)
            continue
        if 245 <= x < 295:
            v = _parse_meta_float(t)
            if v is not None:
                sleep = v
            else:
                journal_parts.append(t)
            continue
        if 295 <= x < 335:
            v = _parse_meta_float(t)
            if v is not None:
                energy = v
            else:
                journal_parts.append(t)
            continue
        if 335 <= x < 368:
            v = _parse_meta_float(t)
            if v is not None:
                stress = v
            else:
                journal_parts.append(t)
            continue
        if 368 <= x < 405 and t in KNOWN_TIME:
            time_of_day = t
            continue
        if 405 <= x < 455 and t in KNOWN_MOOD:
            previous_mood = t
            continue
        if x >= 455:
            tail_blob = (tail_blob + " " + t).strip()
            continue
        journal_parts.append(t)

    if tail_blob:
        fh, rq = _split_face_quality(tail_blob.replace(" ", ""))
        if fh is None:
            fh, rq = _split_face_quality(tail_blob)
        face_hint, reflection_quality = fh, rq

    journal_raw = " ".join(journal_parts)

    ambience = None
    for a in KNOWN_AMBIENCE:
        if re.search(rf"\b{re.escape(a)}\b", journal_raw, flags=re.I):
            ambience = a
            break

    return {
        "id": rid,
        "journal_text": journal_raw,
        "ambience_type": ambience,
        "duration_min": duration,
        "sleep_hours": sleep,
        "energy_level": energy,
        "stress_level": stress,
        "time_of_day": time_of_day,
        "previous_day_mood": previous_mood,
        "face_emotion_hint": face_hint,
        "reflection_quality": reflection_quality,
    }


def parse_test_pdf(
    path: str | Path | None = None,
    *,
    validate: bool = True,
    preprocess: bool = True,
    add_missing_flags: bool = False,
) -> pd.DataFrame:
    """
    Parse PDF test inputs into a dataframe with `FEATURE_COLUMN_ORDER` columns.

    Uses pdfplumber word geometry to separate journal text from tabular metadata.
    """
    pdf_path = Path(path or DEFAULT_TEST_PDF)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Test PDF not found: {pdf_path}")

    rows_out: list[dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            by_top: dict[float, list[dict[str, Any]]] = defaultdict(list)
            for w in words:
                if w["text"] == "id" or "ournal" in w["text"]:
                    continue
                by_top[round(w["top"], 0)].append(w)

            for _top in sorted(by_top.keys()):
                line = sorted(by_top[_top], key=lambda x: x["x0"])
                if not line:
                    continue
                first = line[0]["text"]
                if not first.isdigit() or len(first) != 5:
                    continue
                rid = int(first)
                rows_out.append(_parse_line_words(rid, line[1:]))

    raw = pd.DataFrame(rows_out)
    if preprocess:
        raw = preprocess_feature_dataframe(raw)
    if validate:
        from compassmind.ingestion.schema import validate_features_dataframe

        validate_features_dataframe(raw)
    if add_missing_flags:
        from compassmind.ingestion.preprocess import add_missingness_flags

        raw = add_missingness_flags(raw)

    feat = list(FEATURE_COLUMN_ORDER)
    rest = [c for c in raw.columns if c not in feat]
    return raw.loc[:, feat + rest]
