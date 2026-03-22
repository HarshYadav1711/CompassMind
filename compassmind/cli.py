"""CLI: train (with ablation) and predict to CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from compassmind.features import FeatureConfig
from compassmind.ingestion import load_test_pdf_features, load_training_features
from compassmind.pdf_parse import parse_pdf_rows
from compassmind.predict import predict_dataframe
from compassmind.train_eval import load_bundle, save_bundle, train_bundle

DEFAULT_TRAIN = (
    Path(__file__).resolve().parents[1] / "Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv"
)
DEFAULT_PDF = Path(__file__).resolve().parents[1] / "arvyax_test_inputs_120.xlsx - Sheet1.pdf"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "model_bundle.joblib"


def cmd_train(args: argparse.Namespace) -> None:
    df = load_training_features(args.data, validate=True, add_missing_flags=args.add_missing_flags)
    cfg = FeatureConfig(random_state=args.seed)
    results: dict[str, Any] = {}
    for use_meta in (True, False):
        name = "text_plus_metadata" if use_meta else "text_only"
        bundle = train_bundle(
            df,
            cfg,
            use_metadata=use_meta,
            random_state=args.seed,
            try_xgb_benchmark=args.try_xgb,
            run_stratified_cv=not args.no_cv,
            cv_folds=args.cv_folds,
        )
        out_path = args.artifacts.parent / f"{args.artifacts.stem}_{name}.joblib"
        save_bundle(bundle, out_path)
        results[name] = {
            "metrics": bundle["metrics"],
            "cv_metrics": bundle.get("cv_metrics"),
            "benchmark": bundle.get("benchmark"),
            "backend": bundle.get("backend"),
        }
        print(json.dumps({name: results[name]}, indent=2))
    summary_path = args.artifacts.parent / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    ship = train_bundle(
        df,
        cfg,
        use_metadata=True,
        random_state=args.seed,
        try_xgb_benchmark=args.try_xgb,
        run_stratified_cv=not args.no_cv,
        cv_folds=args.cv_folds,
    )
    save_bundle(ship, args.artifacts)
    print("Saved primary bundle to", args.artifacts)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Load training CSV + test PDF; validate schemas; print summary (no model training)."""
    train = load_training_features(args.data, validate=True, add_missing_flags=args.add_missing_flags)
    test = load_test_pdf_features(args.pdf, validate=True, add_missing_flags=args.add_missing_flags)
    print("training", train.shape, "test", test.shape)
    print("training columns:", list(train.columns))
    print("test columns:", list(test.columns))


def cmd_predict(args: argparse.Namespace) -> None:
    bundle = load_bundle(args.bundle)
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        pdf_path = args.pdf or DEFAULT_PDF
        df = parse_pdf_rows(str(pdf_path))
    out = predict_dataframe(df, bundle)
    out.to_csv(args.out, index=False)
    print("Wrote", args.out, "rows:", len(out))


def main() -> None:
    p = argparse.ArgumentParser(description="CompassMind CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train models and save joblib bundle(s)")
    t.add_argument("--data", type=Path, default=DEFAULT_TRAIN)
    t.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument(
        "--add-missing-flags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align with ingestion missingness flags on structured columns (recommended).",
    )
    t.add_argument(
        "--try-xgb",
        action="store_true",
        help="Benchmark CPU XGBoost vs logistic; keep XGBoost only if val macro-F1 improves meaningfully.",
    )
    t.add_argument("--no-cv", action="store_true", help="Skip stratified K-fold CV reporting (faster).")
    t.add_argument("--cv-folds", type=int, default=5)
    t.set_defaults(func=cmd_train)

    ing = sub.add_parser("ingest", help="Load and validate CSV + PDF (no training)")
    ing.add_argument("--data", type=Path, default=DEFAULT_TRAIN)
    ing.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ing.add_argument("--add-missing-flags", action="store_true")
    ing.set_defaults(func=cmd_ingest)

    pr = sub.add_parser("predict", help="Run inference to predictions.csv")
    pr.add_argument("--bundle", type=Path, default=ARTIFACTS)
    pr.add_argument("--pdf", type=Path, default=None)
    pr.add_argument("--csv", type=Path, default=None)
    pr.add_argument("--out", type=Path, default=Path("predictions.csv"))
    pr.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
