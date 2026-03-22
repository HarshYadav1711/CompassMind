# Edge plan: local, mobile, and offline

## Goals

- Run **fully offline** after dependencies and model artifacts are present.  
- Keep latency predictable on CPU-only hardware (laptops, some tablets).  
- Avoid sending reflections to third-party APIs.

## Packaging

1. **Python environment**  
   - Pin dependencies (`requirements.txt` / `pyproject.toml`).  
   - Ship `artifacts/model_bundle.joblib` next to code or load from app-private storage.

2. **Model artifact size**  
   - TF-IDF + sparse logistic models are modest (tens of MB depending on `max_features`).  
   - If size matters: reduce `max_word_features` / `max_char_features` in `FeatureConfig` and retrain.

## Mobile considerations

- **On-device Python** (e.g., embedded interpreter) is possible but heavy; common patterns:  
  - **Server-on-device** (localhost) via small FastAPI/uvicorn wrapper (optional extra).  
  - **Native port**: export coefficients sparsely or convert to ONNX / Core ML in a later phase (not implemented here).

## Latency

- Inference is dominated by TF-IDF transforms + sparse matrix multiply; typical laptop CPUs handle batch sizes of hundreds in seconds.  
- For interactive UX, batch rows or cap `max_features` after profiling on target hardware.

## Privacy

- No network calls in the default pipeline.  
- Journals stay on disk/memory controlled by the host app; rotate logs and avoid writing raw text to shared caches on mobile.

## Monitoring (offline-friendly)

- Log **distribution shifts** on `uncertain_flag` rate, predicted class histograms, and missing-metadata counts.  
- Periodically retrain when new labeled CSVs arrive.

## Safety

- Rules bias toward **grounding, journaling, light planning, rest** when uncertainty or stress/energy signals demand it.  
- Never present outputs as medical advice; surface uncertainty in the UI when `uncertain_flag` is true.
