"""
Generate evaluation report JSON + refresh ERROR_ANALYSIS.md and EDGE_PLAN.md.

Usage: ``python -m compassmind.evaluation``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from compassmind.evaluation.attribution import block_importance, top_text_features
from compassmind.evaluation.errors import extract_failure_cases
from compassmind.evaluation.holdout import evaluate_holdout
from compassmind.evaluation.metrics import classification_metrics, classification_report_dict
from compassmind.evaluation.robustness import robustness_battery
from compassmind.constants import (
    DEFAULT_TRAINING_CSV,
    EVALUATION_REPORT_JSON,
    PROJECT_ROOT,
    ensure_artifact_dirs,
)
from compassmind.features import FeatureConfig
from compassmind.ingestion import load_training_features
from compassmind.seed import set_global_seed

DEFAULT_REPORT = EVALUATION_REPORT_JSON


def _fmt_float(x: Any, nd: int = 4) -> str:
    if isinstance(x, float):
        if np.isnan(x):
            return "nan"
        return f"{x:.{nd}f}"
    return str(x)


def build_report(
    *,
    data_path: Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    set_global_seed(random_state)
    ensure_artifact_dirs()
    path = data_path or DEFAULT_TRAINING_CSV
    df = load_training_features(path, validate=True, add_missing_flags=True)
    cfg = FeatureConfig(random_state=random_state)

    hr_text = evaluate_holdout(df, cfg, use_metadata=False, random_state=random_state)
    hr_full = evaluate_holdout(df, cfg, use_metadata=True, random_state=random_state)

    def _metrics_block(hr: Any) -> dict[str, Any]:
        le_s, le_i = hr.le_s, hr.le_i
        ms = classification_metrics(
            hr.ys_va,
            hr.pred_state,
            y_proba=hr.proba_state,
            labels=np.arange(len(le_s.classes_)),
        )
        ms["report"] = classification_report_dict(
            hr.ys_va,
            hr.pred_state,
            target_names=[str(x) for x in le_s.classes_],
        )
        mi = classification_metrics(
            hr.yi_va,
            hr.pred_intensity,
            y_proba=hr.proba_intensity,
            labels=np.arange(len(le_i.classes_)),
        )
        mi["report"] = classification_report_dict(
            hr.yi_va,
            hr.pred_intensity,
            target_names=[str(x) for x in le_i.classes_],
        )
        return {"emotional_state": ms, "intensity": mi}

    block_t = _metrics_block(hr_text)
    block_f = _metrics_block(hr_full)
    m_s_text, m_s_full = block_t["emotional_state"], block_f["emotional_state"]
    m_i_text, m_i_full = block_t["intensity"], block_f["intensity"]

    ablation = {
        "text_only": block_t,
        "text_plus_metadata": block_f,
        "delta_macro_f1_state": float(m_s_full["macro_f1"] - m_s_text["macro_f1"]),
        "delta_macro_f1_intensity": float(m_i_full["macro_f1"] - m_i_text["macro_f1"]),
    }

    att_state = block_importance(hr_full.clf_state, n_word=hr_full.n_word, n_char=hr_full.n_char, use_metadata=True)
    att_int = block_importance(hr_full.clf_intensity, n_word=hr_full.n_word, n_char=hr_full.n_char, use_metadata=True)

    w_names = [f"w:{n}" for n in hr_full.wv.get_feature_names_out()]
    c_names = [f"c:{n}" for n in hr_full.cv_vec.get_feature_names_out()]
    n_text = hr_full.n_word + hr_full.n_char
    top_state = top_text_features(
        hr_full.clf_state,
        w_names + c_names,
        top_k=12,
        n_text_features=n_text,
    )

    failures = extract_failure_cases(hr_full, max_cases=12)

    robust = robustness_battery(
        hr_full.val_df,
        wv=hr_full.wv,
        cv_vec=hr_full.cv_vec,
        meta_enc=hr_full.meta_enc,
        use_metadata=True,
        clf_state=hr_full.clf_state,
        le_s=hr_full.le_s,
        n_samples=5,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "n_train_approx": int(len(df) * 0.85),
        "n_val": int(len(hr_full.val_df)),
        "random_state": random_state,
        "ablation": ablation,
        "metrics": {
            "text_only": {
                "emotional_state": {k: v for k, v in m_s_text.items() if k != "report"},
                "intensity": {k: v for k, v in m_i_text.items() if k != "report"},
            },
            "text_plus_metadata": {
                "emotional_state": {k: v for k, v in m_s_full.items() if k != "report"},
                "intensity": {k: v for k, v in m_i_full.items() if k != "report"},
            },
        },
        "per_class_reports": {
            "text_only": {
                "state": m_s_text.get("report", {}),
                "intensity": m_i_text.get("report", {}),
            },
            "text_plus_metadata": {
                "state": m_s_full.get("report", {}),
                "intensity": m_i_full.get("report", {}),
            },
        },
        "feature_attribution_text_plus_metadata": {
            "emotional_state_blocks": {
                "word_l2": att_state.word_l2,
                "char_l2": att_state.char_l2,
                "meta_l2": att_state.meta_l2,
                "word_share": att_state.word_share,
                "char_share": att_state.char_share,
                "meta_share": att_state.meta_share,
                "n_word_features": att_state.n_word,
                "n_char_features": att_state.n_char,
                "n_meta_features": att_state.n_meta,
            },
            "intensity_blocks": {
                "word_l2": att_int.word_l2,
                "char_l2": att_int.char_l2,
                "meta_l2": att_int.meta_l2,
                "word_share": att_int.word_share,
                "char_share": att_int.char_share,
                "meta_share": att_int.meta_share,
            },
            "top_coefficient_magnitude_tokens_state_head": top_state,
        },
        "validation_failure_cases": failures,
        "robustness_scenarios": robust,
    }
    return report


def write_markdown(report: dict[str, Any], path_error: Path, path_edge: Path) -> None:
    """Overwrite ERROR_ANALYSIS.md and EDGE_PLAN.md with data-backed sections."""
    fs = report["feature_attribution_text_plus_metadata"]
    blocks = fs["emotional_state_blocks"]

    # --- ERROR_ANALYSIS.md ---
    lines: list[str] = [
        "# Error analysis and robustness (CompassMind)",
        "",
        "This document is generated from a **stratified holdout** (15%, seed "
        f"{report['random_state']}) that matches training semantics: models are fit **only** on the train split; metrics and cases are from **validation** rows only. It is meant for reviewers and product partners—**not** clinical claims.",
        "",
        "## Summary metrics (validation)",
        "",
        "| Model | emotional_state macro-F1 | emotional_state acc | intensity macro-F1 | intensity acc |",
        "| --- | --- | --- | --- | --- |",
    ]
    mto = report["metrics"]["text_only"]
    mtm = report["metrics"]["text_plus_metadata"]
    lines.append(
        f"| Text-only | {_fmt_float(mto['emotional_state']['macro_f1'])} | {_fmt_float(mto['emotional_state']['accuracy'])} | "
        f"{_fmt_float(mto['intensity']['macro_f1'])} | {_fmt_float(mto['intensity']['accuracy'])} |"
    )
    lines.append(
        f"| Text + metadata | {_fmt_float(mtm['emotional_state']['macro_f1'])} | {_fmt_float(mtm['emotional_state']['accuracy'])} | "
        f"{_fmt_float(mtm['intensity']['macro_f1'])} | {_fmt_float(mtm['intensity']['accuracy'])} |"
    )
    lines.extend(
        [
            "",
            f"**Ablation delta (text+metadata minus text-only):** state macro-F1 **{report['ablation']['delta_macro_f1_state']:+.4f}**, "
            f"intensity macro-F1 **{report['ablation']['delta_macro_f1_intensity']:+.4f}**.",
            "",
            "Interpretation: metadata is **not** guaranteed to help on every split—structured fields can be noisy or missing—but it is retained for product inference where context is often partially available.",
            "",
            "## Text vs metadata contribution (linear state head)",
            "",
            "We aggregate **L2 norm of multinomial logistic coefficients** per block (word TF-IDF, char TF-IDF, metadata stack). This is a faithful linear-model attribution—not SHAP—but it answers “where capacity went” in a defensible baseline.",
            "",
            "| Block | Share of block L2 (state) | Raw block L2 |",
            "| --- | --- | --- |",
            f"| Word n-grams | {_fmt_float(blocks['word_share'], 3)} | {_fmt_float(blocks['word_l2'], 2)} |",
            f"| Char n-grams | {_fmt_float(blocks['char_share'], 3)} | {_fmt_float(blocks['char_l2'], 2)} |",
            f"| Metadata + missingness + OHE | {_fmt_float(blocks['meta_share'], 3)} | {_fmt_float(blocks['meta_l2'], 2)} |",
            "",
            f"Feature counts: **{blocks['n_word_features']}** word, **{blocks['n_char_features']}** char, **{blocks['n_meta_features']}** metadata columns (after encoding).",
            "",
            "### Highest-magnitude token features (state head)",
            "",
        ]
    )
    for tok, mag in fs["top_coefficient_magnitude_tokens_state_head"][:12]:
        lines.append(f"- `{tok}` — {mag:.4f}")

    ib = fs["intensity_blocks"]
    lines.extend(
        [
            "",
            "## Text vs metadata contribution (linear intensity head)",
            "",
            "| Block | Share of block L2 (intensity) | Raw block L2 |",
            "| --- | --- | --- |",
            f"| Word n-grams | {_fmt_float(ib['word_share'], 3)} | {_fmt_float(ib['word_l2'], 2)} |",
            f"| Char n-grams | {_fmt_float(ib['char_share'], 3)} | {_fmt_float(ib['char_l2'], 2)} |",
            f"| Metadata + missingness + OHE | {_fmt_float(ib['meta_share'], 3)} | {_fmt_float(ib['meta_l2'], 2)} |",
            "",
            "## Validation failure cases (real rows)",
            "",
        ]
    )

    for i, case in enumerate(report["validation_failure_cases"][:12], 1):
        summ = case["input_summary"]
        je = summ.get("journal_excerpt", "")[:200]
        lines.extend(
            [
                f"### Case {i} ({case['failure_kind']})",
                "",
                f"- **Journal (excerpt):** {je}",
                f"- **Metadata (subset):** {summ.get('metadata')}",
                f"- **True → pred:** state `{case['true_state']}` → `{case['pred_state']}`; intensity {case['true_intensity']} → {case['pred_intensity']}",
                f"- **What went wrong:** {case['what_went_wrong']}",
                f"- **Why (mechanism):** {case['why_it_failed']}",
                f"- **How to improve:** {case['how_to_improve']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Robustness spot checks (same validation rows, perturbed inputs)",
            "",
            "We re-run the **state** head on controlled variants. **No gold labels** for perturbed text—the point is to show *sensitivity* to known ambiguity patterns (short text, stripped metadata, conflict, typos).",
            "",
        ]
    )
    for r in report["robustness_scenarios"]:
        scen = json.dumps(r["predictions_by_scenario"], indent=2, default=str)
        lines.append(f"- **Row {r['row_index']}** baseline `{r['baseline_state']}` — scenarios:\n\n```json\n{scen}\n```")
    lines.extend(
        [
            "",
            "Expected behavior: ultra-short inputs (`ok`, `fine`) often collapse to a frequent class; missing metadata shifts the decision boundary; contradictory stress vs calm wording exposes softmax’s single-label limitation.",
            "",
            "## Takeaway",
            "",
            "Ambiguous human language will always violate single-label classifiers occasionally. CompassMind pairs **calibrated probabilities**, an explicit **uncertainty layer**, and **conservative** recommendations so the product fails toward safety—not toward false certainty.",
            "",
        ]
    )

    path_error.write_text("\n".join(lines), encoding="utf-8")

    # --- EDGE_PLAN.md ---
    edge = [
        "# Edge deployment plan (CompassMind)",
        "",
        "## Local / offline",
        "",
        "- **Runtime:** Python 3.11+ with pinned `requirements.txt`; no network calls in train/infer by default.",
        "- **Artifacts:** `artifacts/models/model_bundle.joblib` holds sparse TF-IDF vectorizers, `MetadataEncoder`, two calibrated classifiers, label encoders, uncertainty thresholds. Copy alongside application binaries or load from app-private storage.",
        "",
        "## Model size (order of magnitude)",
        "",
        f"- Vocabulary-heavy TF-IDF + sparse matrices: typically **tens of MB** on disk for default `max_word_features=12000` / `max_char_features=8000` (exact size depends on vocabulary).",
        "- Reducing `max_*_features` linearly shrinks both storage and inference cost at the cost of recall on rare n-grams.",
        "",
        "## Latency (CPU, batch size 1)",
        "",
        "- Dominated by: tokenization + TF-IDF transform + sparse matmul + calibration. On a modern laptop, **single-digit to low tens of ms** per row is a reasonable target before optimization; profile on your SKU.",
        "- Batch inference amortizes vectorizer overhead; avoid per-request Python cold start in production services.",
        "",
        "## Tradeoffs",
        "",
        "| Choice | Upside | Downside |",
        "| --- | --- | --- |",
        "| Text + metadata | Context when fields are clean | Extra variance when metadata is wrong/missing |",
        "| Text-only | Simpler, fewer failure modes from bad metadata | Loses structured cues (sleep, stress) |",
        "| Calibrated logistic | Fast, explainable coefs | Ceiling accuracy on ambiguous language |",
        "| Uncertainty + rules | Safe UX under ambiguity | May under-suggest productivity when uncertain |",
        "",
        "## On-device feasibility",
        "",
        "- **Desktop / edge server:** straightforward: same Python stack or package as a small service.",
        "- **Mobile native:** TF-IDF + sparse linear algebra can be ported (e.g., export coefficients + vocabulary); Python-on-device is heavier—prefer a thin native runtime or on-device server.",
        "- **Privacy:** offline inference keeps reflections off third-party APIs; still handle local storage encryption and retention policy in the host app.",
        "",
        "## Monitoring (offline-friendly)",
        "",
        "- Track `uncertain_flag` rate, class histograms, missing-metadata rate, and average journal length. Drift in these without drift in labels usually means upstream capture changed (PDF layout, form fields).",
        "",
    ]
    path_edge.write_text("\n".join(edge), encoding="utf-8")


def main() -> None:
    ensure_artifact_dirs()
    report = build_report()
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown(report, PROJECT_ROOT / "ERROR_ANALYSIS.md", PROJECT_ROOT / "EDGE_PLAN.md")
    print("Wrote", DEFAULT_REPORT)
    print("Updated ERROR_ANALYSIS.md and EDGE_PLAN.md")


if __name__ == "__main__":
    main()
