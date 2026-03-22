# Edge deployment plan (CompassMind)

## Local / offline

- **Runtime:** Python 3.11+ with pinned `requirements.txt`; no network calls in train/infer by default.
- **Artifacts:** `artifacts/models/model_bundle.joblib` holds sparse TF-IDF vectorizers, `MetadataEncoder`, two calibrated classifiers, label encoders, uncertainty thresholds. Copy alongside application binaries or load from app-private storage.

## Model size (order of magnitude)

- Vocabulary-heavy TF-IDF + sparse matrices: typically **tens of MB** on disk for default `max_word_features=12000` / `max_char_features=8000` (exact size depends on vocabulary).
- Reducing `max_*_features` linearly shrinks both storage and inference cost at the cost of recall on rare n-grams.

## Latency (CPU, batch size 1)

- Dominated by: tokenization + TF-IDF transform + sparse matmul + calibration. On a modern laptop, **single-digit to low tens of ms** per row is a reasonable target before optimization; profile on your SKU.
- Batch inference amortizes vectorizer overhead; avoid per-request Python cold start in production services.

## Tradeoffs

| Choice | Upside | Downside |
| --- | --- | --- |
| Text + metadata | Context when fields are clean | Extra variance when metadata is wrong/missing |
| Text-only | Simpler, fewer failure modes from bad metadata | Loses structured cues (sleep, stress) |
| Calibrated logistic | Fast, explainable coefs | Ceiling accuracy on ambiguous language |
| Uncertainty + rules | Safe UX under ambiguity | May under-suggest productivity when uncertain |

## On-device feasibility

- **Desktop / edge server:** straightforward: same Python stack or package as a small service.
- **Mobile native:** TF-IDF + sparse linear algebra can be ported (e.g., export coefficients + vocabulary); Python-on-device is heavier—prefer a thin native runtime or on-device server.
- **Privacy:** offline inference keeps reflections off third-party APIs; still handle local storage encryption and retention policy in the host app.

## Monitoring (offline-friendly)

- Track `uncertain_flag` rate, class histograms, missing-metadata rate, and average journal length. Drift in these without drift in labels usually means upstream capture changed (PDF layout, form fields).
