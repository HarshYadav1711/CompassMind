# CompassMind

## Overview

CompassMind is a local machine learning system designed to move beyond prediction and toward meaningful user guidance.

The system takes short, noisy user reflections along with contextual signals (sleep, stress, energy, time of day) and produces:

* Emotional understanding (state + intensity)
* Actionable recommendation (what to do)
* Timing decision (when to do it)
* Uncertainty-aware outputs

Unlike standard ML pipelines, this system is designed with a **product mindset**, focusing on safety, interpretability, and real-world usability.

---

## Key Design Philosophy

This project follows a simple principle:

> AI should not only understand human states — it should guide users toward better ones.

The system is intentionally designed to:

* handle noisy and imperfect inputs
* avoid overconfident predictions
* provide safe, low-risk recommendations under uncertainty

---

## Approach

### 1. Feature Engineering

Two sources of information are used:

#### Text Features

* TF-IDF (word n-grams)
* TF-IDF (character n-grams)

Character n-grams help handle typos and messy inputs.

#### Metadata Features

* sleep_hours, stress_level, energy_level
* ambience_type, time_of_day
* previous_day_mood, face_emotion_hint
* reflection_quality

---

### 2. Models

#### Emotional State

* Multiclass classification using Logistic Regression / XGBoost

#### Intensity

* Treated as classification (1–5)
* Chosen over regression for stability and discrete outputs

---

### 3. Decision Engine (Core)

A rule-based layer converts predictions into actions.

This layer uses:

* predicted state
* intensity
* stress, energy
* time of day
* uncertainty

The system prioritizes:

* calming interventions under stress
* productivity when energy is high
* rest when energy is low
* safe actions when uncertain

---

### 4. Uncertainty Modeling

Confidence is derived from prediction probabilities.

Uncertainty increases when:

* prediction confidence is low
* top class probabilities are close
* text is very short
* metadata is missing or conflicting

When uncertain, the system defaults to safe recommendations like grounding or light planning.

---

### 5. Ablation Study

| Model           | Observation                                 |
| --------------- | ------------------------------------------- |
| Text-only       | Works for expressive inputs                 |
| Text + Metadata | Strong improvement in short/ambiguous cases |

Metadata plays a critical role when text is weak.

---

## How to Run

**Prerequisites:** Python 3.11+, and the training CSV + test PDF in the project root (filenames as shipped with the assignment).

### 1. Environment

```bash
cd CompassMind
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional: `pip install -e .` so the `compassmind` command is on your `PATH`.

### 2. Train (writes `artifacts/models/model_bundle.joblib`)

```bash
python -m compassmind train --data "Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv"
```

Faster iteration: `python -m compassmind train --no-cv`

### 3. Predict (default: test PDF → `outputs/predictions.csv`)

```bash
python -m compassmind predict
```

Writes **`outputs/predictions.csv`** (the `outputs/` folder is created automatically). Override path: `--out path/to/file.csv`.

Or from a CSV: `python -m compassmind predict --csv your_rows.csv`

### 4. Evaluation report (optional)

```bash
python -m compassmind evaluate
```

Writes `artifacts/reports/evaluation_report.json` and refreshes `ERROR_ANALYSIS.md` / `EDGE_PLAN.md`.

### 5. Ingestion check only (no training)

```bash
python -m compassmind ingest
```

### 6. Tests

```bash
pytest -q
```

### Optional API demo

After `pip install -e ".[demo]"`, train first, then: `uvicorn compassmind.demo_api:app --port 8765`

---

## Output Format

```
id,
predicted_state,
predicted_intensity,
confidence,
uncertain_flag,
what_to_do,
when_to_do
```

---

## Limitations

* Ambiguous short text remains challenging
* Some labels appear noisy or subjective
* Context signals are sometimes incomplete

---

## Future Improvements

* lightweight transformer (on-device)
* better ordinal modeling for intensity
* adaptive decision learning instead of fixed rules

---

## Final Note

The system prioritizes **safe and supportive interventions under uncertainty**, aligning with real-world mental wellness systems.

---
