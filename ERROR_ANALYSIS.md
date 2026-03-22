# Error analysis and failure cases

This document records **representative failure modes** observed or expected for CompassMind on short, noisy reflections. At least ten cases are listed with **cause**, **symptom**, and **mitigation** (local / no hosted APIs).

## 1. Intensity collapse to mid–high bins

**Symptom:** Validation macro-F1 for `intensity` stays modest; some runs predict `4` frequently on the PDF batch.  
**Cause:** Five-way classification with overlapping language cues; ordinal structure is not explicitly modeled.  
**Mitigation:** Prefer reporting intensity with uncertainty; consider ordinal logit or Corn’s method in a follow-up (still local). Optional tree models (e.g., histogram gradient boosting on engineered dense features) are a possible follow-up if they clearly beat logistic regression on time-split CV.

## 2. Metadata OCR gaps in PDF columns

**Symptom:** Rows where numeric columns are missing or merged into journal text; `face_emotion_hint` / `reflection_quality` occasionally unsplit.  
**Cause:** Geometric parsing cannot recover tokens when layout overlaps or ellipsis tokens (`1...`) appear.  
**Mitigation:** Missing-value indicators + `__MISSING__` buckets; parser strips ellipsis tails; unsafe numeric tokens fall back into journal text.

## 3. Contradictory cues (“calm” and “wired” in one reflection)

**Symptom:** Model picks a single `emotional_state` with moderate entropy; `uncertain_flag` may be true.  
**Cause:** Multiclass softmax forces one label; contradictory phrases are common in real journals.  
**Mitigation:** Uncertainty layer + safe actions (grounding, journaling). Product copy should not overclaim a single mood.

## 4. Ambience inferred only from keywords

**Symptom:** `ambience_type` sometimes wrong or `None` if the PDF hides ambience inside garbled tokens.  
**Cause:** Heuristic `\b(ocean|forest|…)\b` search on journal text only.  
**Mitigation:** Missing indicator; model relies more on journal wording when ambience is absent.

## 5. Short reflections (“ok session”)

**Symptom:** High entropy, frequent uncertain flags; intensity unstable.  
**Cause:** Very low token count for both word and char n-grams.  
**Mitigation:** Metadata features (when present) carry signal; rules route to low-risk actions when uncertain.

## 6. Class imbalance toward “neutral / calm” language

**Symptom:** Rare states (e.g., severe overwhelm) under-predicted unless `class_weight='balanced'`.  
**Cause:** Natural frequency skew in user text.  
**Mitigation:** Balanced logistic regression; calibration for probability quality, not for prevalence matching.

## 7. Calibration vs. accuracy trade-off

**Symptom:** Log-loss may move with calibration even when argmax accuracy is flat.  
**Cause:** `CalibratedClassifierCV` refits decision boundaries in probability space.  
**Mitigation:** Track log-loss and macro-F1; thresholds tuned on held-out split for **uncertain rate**, not only accuracy.

## 8. Text-only vs text+metadata ablation flip

**Symptom:** On our validation split, **text-only** sometimes edges out text+metadata for macro-F1 (see `artifacts/ablation_summary.json`).  
**Cause:** With limited data, extra one-hot dimensions can add variance; some metadata fields are noisy or missing.  
**Mitigation:** Ship text+metadata for production when metadata is expected to be present; keep ablation to document the trade-off.

## 9. Decision engine over-uses grounding when `confidence < 0.42`

**Symptom:** Even argmax state looks reasonable, user still sees `grounding_breath_and_senses`.  
**Cause:** Explicit safety rule in `decision.recommend` blending model confidence and human factors.  
**Mitigation:** Tune the `0.42` cutoff alongside clinical / product review; keep separate from `uncertain_flag`.

## 10. PDF row without structured tail columns

**Symptom:** Only journal + `happy_facevague`-style tail; all numerics missing.  
**Cause:** Layout similar to training rows with sparse metadata (valid in training too).  
**Mitigation:** Model trained with missingness; uncertainty typically rises; rules favor safe micro-actions.

## 11. Spurious digits inside words (`organiz4ed`)

**Symptom:** Char n-grams pick up noise; occasionally misleading if digits dominate.  
**Cause:** Deliberate choice not to strip digits from text features.  
**Mitigation:** Word + char fusion; metadata provides orthogonal signal when available.

## 12. Face / quality token glued (`happy_facevague`)

**Symptom:** Parser may miss quality if glued form is nonstandard.  
**Cause:** PDF text extraction merges adjacent tokens.  
**Mitigation:** `_split_face_quality` tries glued and spaced variants; missing flags handle failure.

---

**Review takeaway:** Treat CompassMind as a **decision aid** with explicit uncertainty and conservative actions—not a clinical instrument. External validation on fresh cohorts is required before any real-world deployment.
