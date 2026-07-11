# MAZU Extension — Does a 5-Model Ensemble Fix Overconfidence?

## Motivation

The isotonic-regression calibration experiment (`agent/CALIBRATION_REPORT.md`)
fixed probability *honesty* (Brier score improved substantially) but genuinely
hurt operational alert quality (POD/CSI/HSS at re-derived thresholds got
worse for every hazard). This raised a natural follow-up question: is
calibration-vs-decision-quality an unavoidable trade-off, or was isotonic
regression specifically too aggressive? **Ensemble averaging** (bagging-style:
several independently-trained classifiers, same hyperparameters and training
window, differing only in random seed) is a well-known technique that can
improve both calibration *and* ranking quality simultaneously, since
overconfident individual models often disagree with each other at the
extremes, and averaging naturally moderates that. This was worth testing
directly rather than assumed.

## What it does

`model/12_ensemble.py` trains 5 independent `HistGradientBoostingClassifier`
instances per hazard (`random_state` 42–46, otherwise identical
hyperparameters and the same Jan–Jun 2025 training window as the actual
production models), averages their `predict_proba` output, and evaluates the
average against the **actual saved production model** (not a retrained
stand-in — `agent/saved_models/*.joblib`, loaded read-only) on the same
Jul–Dec 2025 test set, at the **same existing operational thresholds**
(0.50/0.55 — no threshold re-derivation, unlike the calibration experiment,
so this is a direct, fair, apples-to-apples comparison).

This script only reads the production models and writes new files
(`model/ensemble_report.json`, `outputs/ensemble_reliability_diagram.png`);
`agent/saved_models/` is never written to.

## Result: a real but small, inconsistent improvement — not the hoped-for clean win

| Hazard | ROC-AUC (single→ens.) | POD (single→ens.) | CSI (single→ens.) | ECE (single→ens.) | Brier (single→ens.) |
|---|---|---|---|---|---|
| Heatwave | 0.9706 → 0.9704 | 0.849 → **0.841** | 0.552 → 0.553 | 0.039 → 0.036 | 0.030 → 0.029 |
| Flash flood | 0.8732 → 0.8732 | 0.100 → **0.080** | 0.071 → **0.066** | 0.006 → 0.004 | 0.006 → 0.006 |
| Dust storm | 0.887 → **0.893** | 0.535 → **0.527** | 0.121 → 0.119 | 0.085 → **0.087 (worse)** | 0.060 → 0.059 |

**Brier score improved for all 3 hazards** — a genuine, if modest, gain —
confirming ensembling does moderate overconfidence somewhat. But:

- **POD (detection rate) got worse for all 3 hazards**, not better — the
  same direction of trade-off seen with isotonic calibration, just far
  smaller in magnitude (single-digit percentage-point drops here, versus a
  ~55% relative drop for isotonic-calibrated flash flood).
- **ECE (calibration error) actually got slightly *worse* for dust storm**
  (0.085 → 0.087) — ensembling is not a uniform improvement even on the
  metric it is supposed to help most.
- Visually, the before/after reliability diagrams
  (`outputs/ensemble_reliability_diagram.png`) are **nearly indistinguishable**
  from the single-model diagrams — every point still falls well below the
  diagonal. A 5-member ensemble of the same architecture/features/data,
  differing only by random seed, does **not** meaningfully fix the
  overconfidence pattern found in `agent/CALIBRATION_REPORT.md`.

## Why the effect is small: these 5 models are not very diverse

Ensembling helps most when member models make *different kinds* of errors.
Here, all 5 members share the same architecture, the same features, and the
same training data — the only source of diversity is the random seed
controlling `HistGradientBoostingClassifier`'s internal stochasticity
(sample/feature subsampling during boosting). This is a much weaker
diversity source than, e.g., bootstrap-resampling the training data or
mixing different model families — which likely explains why the effect here
is real but small, rather than the dramatic fix isotonic regression achieved
for pure calibration (at a real cost to decision quality).

## Conclusion

**Neither isotonic recalibration nor this same-architecture ensemble cleanly
solves the overconfidence problem without a decision-quality cost.**
Isotonic regression fixes calibration dramatically but hurts POD/CSI/HSS
significantly; this ensemble barely moves calibration and still hurts POD
for every hazard, just by a smaller amount. **Not deployed to production**,
for the same reasons as the calibration experiment (blast radius to ~150
existing verified numbers) plus the added finding here that the expected
benefit doesn't clearly outweigh even that cost. A more diverse ensemble
(bootstrap-resampled data, or mixing model families) is a plausible further
research direction, documented here rather than pursued given time
constraints — the same treatment given to the GNN variant, the neighbor-mean
feature, and the calibration-migration attempt earlier in this project.

## Testing

`model/12b_test_ensemble.py`: confirms the single-model baseline used for
comparison is the actual production model (its ROC-AUC matches the
independently-reported production number in `model_meta.json`, not a
retrained stand-in), confirms the ensemble's seed-42 member reproduces the
production model's ROC-AUC closely (same hyperparameters, same training
window, same seed), a partial bypass that independently retrains a 3-seed
ensemble from raw data and confirms it lands in the same range as the
stored 5-seed result, and explicit, failing-if-wrong assertions locking in
the real findings above (Brier improves everywhere, POD does not, dust
storm's ECE genuinely regresses) rather than leaving them as an eyeballed
chart interpretation.

## Performance impact

None on the model/KG/agent/CAP layers. `12_ensemble.py` only reads the
already-saved production models plus trains temporary, isolated classifier
instances purely for this analysis; `agent/saved_models/` and every existing
test/audit number are unchanged.
