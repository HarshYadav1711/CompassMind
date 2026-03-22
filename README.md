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
- **Uncertainty**: sigmoid-calibrated logistic regression; `uncertain_flag` from state max-probability and normalized entropy (thresholds tuned on a held-out split).  
- **Actions**: deterministic rules in `compassmind/decision.py` (see `EDGE_PLAN.md` for offline/mobile considerations).

See `ERROR_ANALYSIS.md` for failure modes and ablation discussion.
