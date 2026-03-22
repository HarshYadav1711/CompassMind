"""
CompassMind command-line interface.

Commands
--------
``train``       Fit calibrated models; save bundles under ``artifacts/models/``.
``predict``     Run inference; write CSV with required columns.
``evaluate``    Holdout metrics + reports under ``artifacts/reports/``; refresh markdown docs.
``ingest``      Validate CSV + PDF ingestion only.

Entrypoints: ``python -m compassmind`` or ``compassmind`` (if installed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from compassmind.constants import (
    ABLATION_SUMMARY_JSON,
    DEFAULT_MODEL_BUNDLE,
    DEFAULT_PREDICTIONS_CSV,
    DEFAULT_TEST_PDF,
    DEFAULT_TRAINING_CSV,
    EVALUATION_REPORT_JSON,
    PROJECT_ROOT,
    ensure_artifact_dirs,
    ensure_outputs_dir,
)
from compassmind.evaluation.run import build_report, write_markdown
from compassmind.features import FeatureConfig
from compassmind.ingestion import load_test_pdf_features, load_training_features
from compassmind.pdf_parse import parse_pdf_rows
from compassmind.predict import predict_dataframe
from compassmind.seed import set_global_seed
from compassmind.train_eval import load_bundle, save_bundle, train_bundle


def cmd_train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    ensure_artifact_dirs()
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
    ABLATION_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with ABLATION_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote ablation summary to", ABLATION_SUMMARY_JSON)

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


def cmd_evaluate(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    ensure_artifact_dirs()
    report = build_report(data_path=args.data, random_state=args.seed)
    out = args.out or EVALUATION_REPORT_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown(report, PROJECT_ROOT / "ERROR_ANALYSIS.md", PROJECT_ROOT / "EDGE_PLAN.md")
    print("Wrote", out)


def cmd_ingest(args: argparse.Namespace) -> None:
    train = load_training_features(args.data, validate=True, add_missing_flags=args.add_missing_flags)
    test = load_test_pdf_features(args.pdf, validate=True, add_missing_flags=args.add_missing_flags)
    print("training", train.shape, "test", test.shape)
    print("training columns:", list(train.columns))
    print("test columns:", list(test.columns))


def cmd_predict(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    ensure_outputs_dir()
    bundle = load_bundle(args.bundle)
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        pdf_path = args.pdf or DEFAULT_TEST_PDF
        df = parse_pdf_rows(str(pdf_path))
    out = predict_dataframe(df, bundle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("Wrote", args.out, "rows:", len(out))
    print("predicted_intensity distribution:")
    print(out["predicted_intensity"].value_counts().sort_index())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="compassmind",
        description="CompassMind: local reflective-emotion ML + recommendations.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train models; save joblib bundle(s) to artifacts/models/")
    t.add_argument("--data", type=Path, default=DEFAULT_TRAINING_CSV)
    t.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_MODEL_BUNDLE,
        help="Primary bundle path (ablation variants share the same stem).",
    )
    t.add_argument("--seed", type=int, default=42, help="Global + sklearn random_state (reproducibility).")
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

    ev = sub.add_parser("evaluate", help="Evaluation report + ERROR_ANALYSIS.md / EDGE_PLAN.md")
    ev.add_argument("--data", type=Path, default=DEFAULT_TRAINING_CSV)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--out", type=Path, default=None, help=f"Default: {EVALUATION_REPORT_JSON}")
    ev.set_defaults(func=cmd_evaluate)

    ing = sub.add_parser("ingest", help="Load and validate CSV + PDF (no training)")
    ing.add_argument("--data", type=Path, default=DEFAULT_TRAINING_CSV)
    ing.add_argument("--pdf", type=Path, default=DEFAULT_TEST_PDF)
    ing.add_argument("--add-missing-flags", action="store_true")
    ing.set_defaults(func=cmd_ingest)

    pr = sub.add_parser("predict", help="Inference: write predictions CSV")
    pr.add_argument("--bundle", type=Path, default=DEFAULT_MODEL_BUNDLE)
    pr.add_argument("--pdf", type=Path, default=None)
    pr.add_argument("--csv", type=Path, default=None)
    pr.add_argument("--out", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    pr.add_argument("--seed", type=int, default=42, help="NumPy RNG for any stochastic post-steps.")
    pr.set_defaults(func=cmd_predict)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
