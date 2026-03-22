"""Text handling: preserve typos and noise; only minimal whitespace cleanup."""

from __future__ import annotations

import re
import unicodedata


_WS_RE = re.compile(r"\s+")


def light_normalize(text: str) -> str:
    """Collapse repeated whitespace; keep casing and punctuation."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = _WS_RE.sub(" ", t).strip()
    return t


def journal_for_features(text: str) -> str:
    """Feature string for vectorizers (same as light_normalize)."""
    return light_normalize(text)
