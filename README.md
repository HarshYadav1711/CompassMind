# CompassMind

Production-style **local** pipeline for the Arvyax reflective emotion task: multiclass `emotional_state`, discrete `intensity` (1–5), calibrated uncertainty, and a **rule-based** `what_to_do` / `when_to_do` layer that defaults to safe actions when confidence is low.

## Stack

- Python 3.11+, pandas, NumPy, scikit-learn, joblib, pydantic, pytest  
- PDF test inputs: `pdfplumber` (with `pypdf` available for alternate parsing)  
- **No** cloud APIs, paid keys, or hosted LLMs

## Data

- Train: `Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv`  
- Test PDF: `arvyax_test_inputs_120.xlsx - Sheet1.pdf`

## Data ingestion (no training)

The `compassmind.ingestion` package loads the CSV and PDF, applies **deterministic preprocessing** (lowercase journal text, trim, Unicode NFKC, standardized missing values, optional `{col}_is_missing` flags), and validates each row with **strict Pydantic schemas** (`FeatureRowStrict`, `TrainingRowStrict`).

```bash
set PYTHONPATH=%CD%
python -m compassmind.cli ingest
```

Programmatic use:

```python
from compassmind.ingestion import load_training_features, load_test_pdf_features

train_df = load_training_features()
test_df = load_test_pdf_features()
```

## Baseline training (features + calibrated models)

- **Text**: word TF-IDF (n-grams via `word_ngram_range`) + character WB TF-IDF (`char_ngram_range`).
- **Metadata**: numeric median impute + `StandardScaler`; 9 explicit `{col}_is_missing` flags; categorical one-hot with `__MISSING__`.
- **Targets**: `emotional_state` (multiclass) and `intensity` (1–5 as 5-way classification).
- **Models**: sigmoid-calibrated `LogisticRegression` (default). Optional `--try-xgb` benchmarks **CPU** `XGBClassifier` and switches only if validation macro-F1 improves meaningfully.
- **Ablation**: `train` fits **text-only** vs **text+metadata** and writes `ablation_summary.json`.
- **Validation**: stratified holdout + optional stratified K-fold metrics (`--no-cv` to skip CV for speed).

```bash
set PYTHONPATH=%CD%
python -m compassmind.cli train --data "Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv"
python -m compassmind.cli train --try-xgb --no-cv
```

Artifacts: `artifacts/model_bundle.joblib` (vectorizers, `MetadataEncoder`, calibrated `clf_state` / `clf_intensity`, label encoders, uncertainty thresholds) plus `*.metrics.json`.

## Quickstart

```bash
cd CompassMind
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
set PYTHONPATH=%CD%
python -m compassmind.cli train --data "Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv"
python -m compassmind.cli predict --out predictions.csv
```

Outputs:

- `artifacts/model_bundle.joblib` — fitted vectorizers, metadata encoder, calibrated models, thresholds  
- `artifacts/ablation_summary.json` — text-only vs text+metadata validation metrics  
- `predictions.csv` — columns: `id`, `predicted_state`, `predicted_intensity`, `confidence`, `uncertain_flag`, `what_to_do`, `when_to_do`

## Tests

```bash
set PYTHONPATH=%CD%
pytest -q
```

## Design notes

- **Text**: ingestion lowercases journal text and trims whitespace (typos/digits preserved); modeling uses word + char TF-IDF on that preprocessed field.  
- **Metadata**: numeric imputation + scaling + missing flags; categorical one-hot with `__MISSING__`.  
- **Uncertainty** (`compassmind/uncertainty.py`): **confidence** = weighted blend of max calibrated probabilities (state + intensity). **`uncertain_flag`** ∈ {0,1} is rule-based OR of: low confidence, high entropy, small top-1 vs top-2 margin, short/weak journal text, sparse metadata, or a few conservative “signal conflict” checks. Training still stores tuned `conf_thresh` / `ent_thresh` in the bundle for the probability/entropy slice.  
- **Recommendations** (`compassmind/decision.py`): transparent mapping from `predicted_state`, `predicted_intensity`, stress, energy, time of day, and uncertainty to **`what_to_do`** / **`when_to_do`** using a fixed vocabulary (`box_breathing`, `journaling`, `grounding`, `deep_work`, `rest`, `light_planning`, `movement`, `pause`, …). Under uncertainty, defaults favor **breathing / grounding / pause** over aggressive productivity.  
- **Offline**: see `EDGE_PLAN.md`.

See `ERROR_ANALYSIS.md` for failure modes and ablation discussion.
