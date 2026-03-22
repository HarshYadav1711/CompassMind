"""
Repository paths and artifact locations (single source of truth for CLI and docs).

Layout::

    artifacts/
      models/          # trained bundles (*.joblib)
      reports/         # evaluation_report.json, ablation_summary.json
    outputs/           # prediction CSVs and other run outputs
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"

# Primary shipped model (text + metadata)
DEFAULT_MODEL_BUNDLE: Path = MODELS_DIR / "model_bundle.joblib"

# Evaluation / ablation outputs
EVALUATION_REPORT_JSON: Path = REPORTS_DIR / "evaluation_report.json"
ABLATION_SUMMARY_JSON: Path = REPORTS_DIR / "ablation_summary.json"

# Bundled datasets (filenames as provided for the assignment)
DEFAULT_TRAINING_CSV: Path = PROJECT_ROOT / "Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv"
DEFAULT_TEST_PDF: Path = PROJECT_ROOT / "arvyax_test_inputs_120.xlsx - Sheet1.pdf"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
DEFAULT_PREDICTIONS_CSV: Path = OUTPUTS_DIR / "predictions.csv"


def ensure_artifact_dirs() -> None:
    """Create artifact subdirectories if missing."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_outputs_dir() -> None:
    """Create the outputs directory (predictions CSV default location)."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
