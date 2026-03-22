from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

EmotionalState = Literal["calm", "focused", "mixed", "neutral", "overwhelmed", "restless"]
TimeOfDay = Literal["morning", "afternoon", "evening", "night", "early_morning"]
AmbienceType = Literal["ocean", "forest", "mountain", "rain", "cafe"]
FaceHint = Literal["calm_face", "tired_face", "tense_face", "happy_face", "neutral_face", "none"]
ReflectionQuality = Literal["clear", "vague", "conflicted"]
PreviousMood = Literal["calm", "mixed", "neutral", "overwhelmed", "restless", "focused"]


class ReflectionInput(BaseModel):
    """One user reflection row (features used at inference)."""

    id: int
    journal_text: str = Field(default="", description="Raw or lightly normalized journal; keep noise.")
    ambience_type: Optional[str] = None
    duration_min: Optional[float] = None
    sleep_hours: Optional[float] = None
    energy_level: Optional[float] = None
    stress_level: Optional[float] = None
    time_of_day: Optional[str] = None
    previous_day_mood: Optional[str] = None
    face_emotion_hint: Optional[str] = None
    reflection_quality: Optional[str] = None


class PredictionRow(BaseModel):
    """CSV output row."""

    id: int
    predicted_state: str
    predicted_intensity: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertain_flag: bool
    what_to_do: str
    when_to_do: str
    supportive_message: Optional[str] = Field(
        default=None,
        description="Optional; not written to assignment CSV unless requested.",
    )
