# Error analysis and robustness (CompassMind)

This document is generated from a **stratified holdout** (15%, seed 42) that matches training semantics: models are fit **only** on the train split; metrics and cases are from **validation** rows only. It is meant for reviewers and product partners—**not** clinical claims.

## Summary metrics (validation)

| Model | emotional_state macro-F1 | emotional_state acc | intensity macro-F1 | intensity acc |
| --- | --- | --- | --- | --- |
| Text-only | 0.6465 | 0.6444 | 0.1646 | 0.2611 |
| Text + metadata | 0.6388 | 0.6389 | 0.1440 | 0.2278 |

**Ablation delta (text+metadata minus text-only):** state macro-F1 **-0.0077**, intensity macro-F1 **-0.0207**.

Interpretation: metadata is **not** guaranteed to help on every split—structured fields can be noisy or missing—but it is retained for product inference where context is often partially available.

## Text vs metadata contribution (linear state head)

We aggregate **L2 norm of multinomial logistic coefficients** per block (word TF-IDF, char TF-IDF, metadata stack). This is a faithful linear-model attribution—not SHAP—but it answers “where capacity went” in a defensible baseline.

| Block | Share of block L2 (state) | Raw block L2 |
| --- | --- | --- |
| Word n-grams | 0.457 | 13.37 |
| Char n-grams | 0.446 | 13.03 |
| Metadata + missingness + OHE | 0.097 | 2.85 |

Feature counts: **1412** word, **3783** char, **39** metadata columns (after encoding).

### Highest-magnitude token features (state head)

- `w:nothing` — 1.3942
- `w:but not` — 1.2657
- `w:but` — 1.2216
- `w:felt more` — 1.1259
- `w:more` — 1.1101
- `w:drained` — 1.0755
- `w:felt distracted` — 1.0698
- `w:my` — 1.0544
- `w:much` — 1.0181
- `w:felt mentally` — 1.0059
- `w:clearer` — 0.9888
- `w:nothing really` — 0.9646

## Text vs metadata contribution (linear intensity head)

| Block | Share of block L2 (intensity) | Raw block L2 |
| --- | --- | --- |
| Word n-grams | 0.505 | 14.13 |
| Char n-grams | 0.409 | 11.45 |
| Metadata + missingness + OHE | 0.085 | 2.39 |

## Validation failure cases (real rows)

### Case 1 (state)

- **Journal (excerpt):** ended up locked in for a bit. ocean audio was nice. not sure why but it shifted.
- **Metadata (subset):** {'stress': 1.0, 'energy': 3.0, 'time_of_day': 'afternoon', 'face': None}
- **True → pred:** state `focused` → `overwhelmed`; intensity 1 → 4
- **What went wrong:** Predicted `overwhelmed` instead of labeled `focused` (intensity 4 vs 1).
- **Why (mechanism):** Multiclass softmax must pick one label; overlapping vocabulary across states (e.g. tired vs overwhelmed) creates near ties.
- **How to improve:** Add contrastive / hard-negative pairs for commonly confused state pairs. Try ordinal intensity (e.g. cumulative link) or cost-sensitive loss on adjacent bins. Use `uncertain_flag` + conservative UX whenever margin or entropy is poor (already in pipeline).

### Case 2 (state)

- **Journal (excerpt):** still anxious a bit
- **Metadata (subset):** {'stress': 5.0, 'energy': 5.0, 'time_of_day': 'morning', 'face': 'neutral_face'}
- **True → pred:** state `restless` → `neutral`; intensity 1 → 1
- **What went wrong:** Predicted `neutral` instead of labeled `restless` (intensity 1 vs 1).
- **Why (mechanism):** Bag-of-ngrams underfits when the journal has almost no tokens. Multiclass softmax must pick one label; overlapping vocabulary across states (e.g. tired vs overwhelmed) creates near ties.
- **How to improve:** Down-weight or abstain on ultra-short reflections in evaluation and product. Add contrastive / hard-negative pairs for commonly confused state pairs. Try ordinal intensity (e.g. cumulative link) or cost-sensitive loss on adjacent bins.

### Case 3 (state)

- **Journal (excerpt):** honestly honestly not much change.
- **Metadata (subset):** {'stress': 1.0, 'energy': 4.0, 'time_of_day': 'evening', 'face': 'tired_face'}
- **True → pred:** state `focused` → `restless`; intensity 1 → 4
- **What went wrong:** Predicted `restless` instead of labeled `focused` (intensity 4 vs 1).
- **Why (mechanism):** Bag-of-ngrams underfits when the journal has almost no tokens. Multiclass softmax must pick one label; overlapping vocabulary across states (e.g. tired vs overwhelmed) creates near ties.
- **How to improve:** Down-weight or abstain on ultra-short reflections in evaluation and product. Add contrastive / hard-negative pairs for commonly confused state pairs. Try ordinal intensity (e.g. cumulative link) or cost-sensitive loss on adjacent bins.

### Case 4 (state)

- **Journal (excerpt):** at first felt good for a moment.
- **Metadata (subset):** {'stress': 5.0, 'energy': 4.0, 'time_of_day': 'afternoon', 'face': None}
- **True → pred:** state `restless` → `overwhelmed`; intensity 5 → 3
- **What went wrong:** Predicted `overwhelmed` instead of labeled `restless` (intensity 3 vs 5).
- **Why (mechanism):** Bag-of-ngrams underfits when the journal has almost no tokens. Multiclass softmax must pick one label; overlapping vocabulary across states (e.g. tired vs overwhelmed) creates near ties.
- **How to improve:** Down-weight or abstain on ultra-short reflections in evaluation and product. Add contrastive / hard-negative pairs for commonly confused state pairs. Try ordinal intensity (e.g. cumulative link) or cost-sensitive loss on adjacent bins.

### Case 5 (state)

- **Journal (excerpt):** by the end felt good for a moment.
- **Metadata (subset):** {'stress': 2.0, 'energy': 3.0, 'time_of_day': 'morning', 'face': None}
- **True → pred:** state `focused` → `overwhelmed`; intensity 5 → 3
- **What went wrong:** Predicted `overwhelmed` instead of labeled `focused` (intensity 3 vs 5).
- **Why (mechanism):** Bag-of-ngrams underfits when the journal has almost no tokens. Multiclass softmax must pick one label; overlapping vocabulary across states (e.g. tired vs overwhelmed) creates near ties.
- **How to improve:** Down-weight or abstain on ultra-short reflections in evaluation and product. Add contrastive / hard-negative pairs for commonly confused state pairs. Try ordinal intensity (e.g. cumulative link) or cost-sensitive loss on adjacent bins.

### Case 6 (state)

- **Journal (excerpt):** i guess back to normal after.
- **Metadata (subset):** {'stress': 5.0, 'energy': 4.0, 'time_of_day': 'night', 'face': 'happy_face'}
- **True → pred:** state `neutral` → `restless`; intensity 2 → 4
- **What went wrong:** Predicted `restless` instead of labeled `neutral` (intensity 4 vs 2).
- **Why (mechanism):** Bag-of-ngrams underfits when the journal has almost no tokens. Multiclass softmax must pick one label; overlapping vocabulary across states (e.g. tired vs overwhelmed) creates near ties.
- **How to improve:** Down-weight or abstain on ultra-short reflections in evaluation and product. Add contrastive / hard-negative pairs for commonly confused state pairs. Try ordinal intensity (e.g. cumulative link) or cost-sensitive loss on adjacent bins.

### Case 7 (state)

- **Journal (excerpt):** during the session mind was all over the place. later it changed felt better after a bit.
- **Metadata (subset):** {'stress': 2.0, 'energy': 5.0, 'time_of_day': 'afternoon', 'face': 'neutral_face'}
- **True → pred:** state `calm` → `overwhelmed`; intensity 5 → 1
- **What went wrong:** Predicted `overwhelmed` instead of labeled `calm` (intensity 1 vs 5).
- **Why (mechanism):** Multiclass softmax must pick one label; overlapping vocabulary across states (e.g. tired vs overwhelmed) creates near ties.
- **How to improve:** Add contrastive / hard-negative pairs for commonly confused state pairs. Try ordinal intensity (e.g. cumulative link) or cost-sensitive loss on adjacent bins. Use `uncertain_flag` + conservative UX whenever margin or entropy is poor (already in pipeline).

### Case 8 (state)

- **Journal (excerpt):** still heavy
- **Metadata (subset):** {'stress': 3.0, 'energy': 5.0, 'time_of_day': 'afternoon', 'face': None}
- **True → pred:** state `focused` → `overwhelmed`; intensity 2 → 4
- **What went wrong:** Predicted `overwhelmed` instead of labeled `focused` (intensity 4 vs 2).
- **Why (mechanism):** Bag-of-ngrams underfits when the journal has almost no tokens. Multiclass softmax must pick one label; overlapping vocabulary across states (e.g. tired vs overwhelmed) creates near ties.
- **How to improve:** Down-weight or abstain on ultra-short reflections in evaluation and product. Add contrastive / hard-negative pairs for commonly confused state pairs. Try ordinal intensity (e.g. cumulative link) or cost-sensitive loss on adjacent bins.

### Case 9 (state)

- **Journal (excerpt):** not sure what changed
- **Metadata (subset):** {'stress': 5.0, 'energy': 5.0, 'time_of_day': 'evening', 'face': 'tired_face'}
- **True → pred:** state `mixed` → `calm`; intensity 1 → 1
- **What went wrong:** Predicted `calm` instead of labeled `mixed` (intensity 1 vs 1).
- **Why (mechanism):** Bag-of-ngrams underfits when the journal has almost no tokens. Multiclass softmax must pick one label; overlapping vocabulary across states (e.g. tired vs overwhelmed) creates near ties.
- **How to improve:** Down-weight or abstain on ultra-short reflections in evaluation and product. Add contrastive / hard-negative pairs for commonly confused state pairs. Try ordinal intensity (e.g. cumulative link) or cost-sensitive loss on adjacent bins.

### Case 10 (state)

- **Journal (excerpt):** at first that helped a little.
- **Metadata (subset):** {'stress': 2.0, 'energy': 1.0, 'time_of_day': 'morning', 'face': None}
- **True → pred:** state `neutral` → `calm`; intensity 4 → 3
- **What went wrong:** Predicted `calm` instead of labeled `neutral` (intensity 3 vs 4).
- **Why (mechanism):** Bag-of-ngrams underfits when the journal has almost no tokens. Multiclass softmax must pick one label; overlapping vocabulary across states (e.g. tired vs overwhelmed) creates near ties.
- **How to improve:** Down-weight or abstain on ultra-short reflections in evaluation and product. Add contrastive / hard-negative pairs for commonly confused state pairs. Try ordinal intensity (e.g. cumulative link) or cost-sensitive loss on adjacent bins.

### Case 11 (state)

- **Journal (excerpt):** lowkey felt pretty grounded. i stayed with it anyway.
- **Metadata (subset):** {'stress': 4.0, 'energy': 2.0, 'time_of_day': 'night', 'face': 'happy_face'}
- **True → pred:** state `calm` → `neutral`; intensity 5 → 4
- **What went wrong:** Predicted `neutral` instead of labeled `calm` (intensity 4 vs 5).
- **Why (mechanism):** Bag-of-ngrams underfits when the journal has almost no tokens. Multiclass softmax must pick one label; overlapping vocabulary across states (e.g. tired vs overwhelmed) creates near ties.
- **How to improve:** Down-weight or abstain on ultra-short reflections in evaluation and product. Add contrastive / hard-negative pairs for commonly confused state pairs. Try ordinal intensity (e.g. cumulative link) or cost-sensitive loss on adjacent bins.

### Case 12 (state)

- **Journal (excerpt):** for some reason kinda calm now.
- **Metadata (subset):** {'stress': 5.0, 'energy': 4.0, 'time_of_day': 'evening', 'face': None}
- **True → pred:** state `focused` → `overwhelmed`; intensity 2 → 4
- **What went wrong:** Predicted `overwhelmed` instead of labeled `focused` (intensity 4 vs 2).
- **Why (mechanism):** Bag-of-ngrams underfits when the journal has almost no tokens. Multiclass softmax must pick one label; overlapping vocabulary across states (e.g. tired vs overwhelmed) creates near ties. Calm wording coexists with high stress metadata; the model cannot represent 'both' in one class.
- **How to improve:** Down-weight or abstain on ultra-short reflections in evaluation and product. Add contrastive / hard-negative pairs for commonly confused state pairs. Try ordinal intensity (e.g. cumulative link) or cost-sensitive loss on adjacent bins.

## Robustness spot checks (same validation rows, perturbed inputs)

We re-run the **state** head on controlled variants. **No gold labels** for perturbed text—the point is to show *sensitivity* to known ambiguity patterns (short text, stripped metadata, conflict, typos).

- **Row 15** baseline `focused` — scenarios:

```json
{
  "baseline": "focused",
  "text_ok": "calm",
  "text_fine": "calm",
  "text_missing_meta": "overwhelmed",
  "text_conflict_high_stress_calm_words": "mixed",
  "text_typo_heavy": "focused"
}
```
- **Row 77** baseline `calm` — scenarios:

```json
{
  "baseline": "calm",
  "text_ok": "neutral",
  "text_fine": "calm",
  "text_missing_meta": "overwhelmed",
  "text_conflict_high_stress_calm_words": "mixed",
  "text_typo_heavy": "calm"
}
```
- **Row 78** baseline `restless` — scenarios:

```json
{
  "baseline": "restless",
  "text_ok": "calm",
  "text_fine": "focused",
  "text_missing_meta": "overwhelmed",
  "text_conflict_high_stress_calm_words": "mixed",
  "text_typo_heavy": "restless"
}
```
- **Row 116** baseline `mixed` — scenarios:

```json
{
  "baseline": "mixed",
  "text_ok": "neutral",
  "text_fine": "neutral",
  "text_missing_meta": "mixed",
  "text_conflict_high_stress_calm_words": "mixed",
  "text_typo_heavy": "neutral"
}
```
- **Row 136** baseline `calm` — scenarios:

```json
{
  "baseline": "calm",
  "text_ok": "calm",
  "text_fine": "calm",
  "text_missing_meta": "calm",
  "text_conflict_high_stress_calm_words": "mixed",
  "text_typo_heavy": "calm"
}
```

Expected behavior: ultra-short inputs (`ok`, `fine`) often collapse to a frequent class; missing metadata shifts the decision boundary; contradictory stress vs calm wording exposes softmax’s single-label limitation.

## Takeaway

Ambiguous human language will always violate single-label classifiers occasionally. CompassMind pairs **calibrated probabilities**, an explicit **uncertainty layer**, and **conservative** recommendations so the product fails toward safety—not toward false certainty.
