"""
Optional local demo API (FastAPI + uvicorn).

Run (after `pip install fastapi uvicorn`):
  set PYTHONPATH=%CD%
  uvicorn compassmind.demo_api:app --reload --port 8765

POST /predict_json with a JSON body matching `ReflectionInput` fields (see `compassmind.schemas`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel

try:
    from fastapi import FastAPI
except ImportError as e:  # pragma: no cover
    raise ImportError("Install optional deps: pip install fastapi uvicorn") from e

from compassmind.predict import predict_dataframe
from compassmind.schemas import ReflectionInput
from compassmind.train_eval import load_bundle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "artifacts" / "model_bundle.joblib"

_bundle = load_bundle(DEFAULT_BUNDLE)
app = FastAPI(title="CompassMind local demo", version="0.1.0")


class PredictResponse(BaseModel):
    id: int
    predicted_state: str
    predicted_intensity: int
    confidence: float
    uncertain_flag: bool
    what_to_do: str
    when_to_do: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict_json", response_model=PredictResponse)
def predict_json(body: ReflectionInput) -> PredictResponse:
    df = pd.DataFrame([body.model_dump()])
    out = predict_dataframe(df, _bundle).iloc[0].to_dict()
    return PredictResponse(**out)
