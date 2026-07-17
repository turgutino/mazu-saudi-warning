# MAZU Extension — POD/FAR/CSI/HSS: WMO-Standard Verification Metrics

## Motivation

POD (Probability of Detection), FAR (False Alarm Ratio), CSI (Critical
Success Index), and HSS (Heidke Skill Score) are the standard
contingency-table metrics national meteorological services (WMO, NCM) use
to verify operational forecasts. Our own evaluation (ROC-AUC/PR-AUC) is
methodologically correct for rare-event classification and threshold-
independent, but doesn't speak the same operational language a meteorology-
background reviewer expects. This extension adds POD/FAR/CSI/HSS **alongside**
the existing metrics — not replacing them.

## What it does

`model/08_meteorological_metrics.py` computes these 4 metrics for all 3
hazards, at each hazard's own **operational threshold** — deliberately reused
from `DetectionEngine.RULES[hazard]["severity"][1][1]` (0.50 for flash_flood,
0.55 for heatwave/dust_storm), the exact same threshold already used
elsewhere in this codebase for CAP alert severity — rather than inventing a
second, independent cutoff scale.

Critically, this script **does not retrain any model**. It loads the
already-saved, already-verified production `.joblib` models and rebuilds
the exact same held-out Jul-Dec test set each was originally verified
against, using the identical code paths as `agent/01_train_and_save_models.py`
and `model/07_dust_storm_forecast.py`. This means there is zero risk of these
new numbers reflecting a different model than the one the agent actually
serves.

Results are merged into `agent/saved_models/model_meta.json` (adding a
`meteorological_metrics` field per hazard, alongside the existing
`roc_auc`/`pr_auc` — nothing is overwritten), and `forecast_tool` now
surfaces this field on every call.

## A real, disclosed finding: threshold-dependent metrics tell a different story than ROC-AUC

| Hazard | ROC-AUC | POD | FAR | CSI | HSS |
|---|---|---|---|---|---|
| heatwave | 0.971 | 0.849 | 0.388 | 0.552 | 0.693 |
| flash_flood | 0.873 | **0.100** | 0.803 | 0.071 | 0.130 |
| dust_storm | 0.887 | 0.535 | 0.865 | 0.121 | 0.190 |

flash_flood has a strong, verified ROC-AUC (0.873, threshold-independent —
the model ranks true flash-flood cells above non-flash-flood cells well
across ALL possible thresholds). But at its fixed 0.50 operational
threshold, POD is only 0.10 — the model correctly flags just 10% of true
flash-flood cells at that specific cutoff, and 80% of its alerts at that
cutoff are false alarms (FAR=0.80).

This is not a contradiction or a bug: it's the expected behavior for an
**extremely rare event** (flash_flood positive rate in the test set is
~0.5%) evaluated at a single fixed threshold, and it is exactly the finding
POD/FAR/CSI/HSS exist to surface that ROC-AUC alone cannot. It is reported
here honestly rather than only showing the more flattering ROC-AUC number —
consistent with this project's standing practice of disclosing real
limitations. dust_storm shows a milder version of the same pattern
(FAR=0.865 at threshold 0.55); heatwave, whose positive rate is far higher,
shows much healthier scores (POD=0.849, CSI=0.552) at the same style of
threshold, confirming the pattern tracks base rate as expected, not a
methodology error specific to one hazard.

## Testing

9 new checks in `02_test_tools.py` (143/143 total, up from 134): exact
match between `forecast_tool`'s live `meteorological_metrics` field and
`model_meta.json`'s stored values (a bypass, re-read from the source dict
directly) for all 3 hazards, a range check that all 4 scores are valid
[0,1] values, and an explicit assertion that flash_flood's POD is genuinely
low (locking in the real finding above as expected behavior, not tolerating
it as an unexplained anomaly).

`FULL_SYSTEM_AUDIT.py` Section M (9 checks, 99/99 total) is a full
independent bypass: it reloads the raw dataset, rebuilds the flash_flood
test set via the same model-building module, loads the saved `.joblib`
model, predicts, and recomputes POD/FAR/CSI from a fresh confusion matrix —
entirely independent of both `tools.py` and `08_meteorological_metrics.py`'s
own stored numbers — and confirms they match within float tolerance. It also
re-confirms the low-POD finding a 2nd time and checks all 3 hazards' metrics
are finite (no silent NaN from a degenerate confusion matrix).

## Performance impact

None on the model/KG/CAP/detection layers. `08_meteorological_metrics.py`
only reads the already-trained models to produce evaluation numbers;
`forecast_tool` only adds one more field (a dict lookup) to its return
value. All 143 unit tests and all 99 audit checks (covering every prior
extension, not just this one) pass unchanged.
