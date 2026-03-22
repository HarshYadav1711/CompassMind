# CompassMind

Production-style **local** pipeline for the Arvyax reflective emotion task: multiclass `emotional_state`, discrete `intensity` (1–5), calibrated uncertainty, and a **rule-based** `what_to_do` / `when_to_do` layer that defaults to safe actions when confidence is low.

## Stack

- Python 3.11+, pandas, NumPy, scikit-learn, joblib, pydantic, pytest  
- PDF test inputs: `pdfplumber` (with `pypdf` available for alternate parsing)  
- **No** cloud APIs, paid keys, or hosted LLMs

## Data

- Train: `Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv`  
- Test PDF: `arvyax_test_inputs_120.xlsx - Sheet1.pdf`

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

- **Text**: word + character TF-IDF; light whitespace normalization only (typos preserved).  
- **Metadata**: numeric imputation + scaling + missing flags; categorical one-hot with `__MISSING__`.  
- **Uncertainty**: sigmoid-calibrated logistic regression; `uncertain_flag` from state max-probability and normalized entropy (thresholds tuned on a held-out split).  
- **Actions**: deterministic rules in `compassmind/decision.py` (see `EDGE_PLAN.md` for offline/mobile considerations).

See `ERROR_ANALYSIS.md` for failure modes and ablation discussion.
