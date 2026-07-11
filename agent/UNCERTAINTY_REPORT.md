# MAZU Extension — Ensemble Uncertainty Quantification (deployed)

## Motivation

Two prior experiments this cycle (`agent/CALIBRATION_REPORT.md`,
`agent/ENSEMBLE_REPORT.md`) both tried to *fix* the genuine overconfidence
found in the reliability-diagram audit (`agent/CALIBRATION_REPORT.md`) by
changing the point estimate itself — isotonic recalibration and
ensemble-averaging. Both were tested rigorously and both were **rejected**:
either the point estimate got more honest but decision quality (POD/CSI/HSS)
got measurably worse, or the calibration gain was too small to justify any
change at all. Neither is deployed to production.

This raised the real question behind the brainstorm for a new tool: if the
point estimate can't be safely changed, is there still something honest and
useful to say about *how much to trust it* — without touching it? The 5-model
ensemble already trained for `agent/ENSEMBLE_REPORT.md` (`model/13_train_ensemble_members.py`,
5 independently-seeded `HistGradientBoostingClassifier` instances per hazard,
same architecture/features/training window as production) was previously only
used to compute an *averaged* probability. Its **spread** across the 5 members
had never been surfaced — and disagreement between 5 otherwise-identical
models on the same input is itself a genuine, cheap, honest signal: it is
large exactly where the models are guessing on unfamiliar territory, and small
where they agree. That is the "elegant" resolution actually pursued here: add
an honest uncertainty *context* field alongside the unchanged production
probability, rather than trying to alter the probability itself.

## What it does

`agent/tools.py`'s `forecast_tool` now returns a 4th top-level field,
`uncertainty`, computed by `_ensemble_uncertainty(hazard, X)`:

```python
{
  "mean":      # mean of the 5 ensemble members' predict_proba on this exact input
  "std":       # standard deviation across the 5 members -- the disagreement signal
  "range":     # [min, max] across the 5 members
  "n_members": 5
}
```

The 5 members per hazard are the same `agent/saved_models/ensemble/{hazard}_seed{42..46}.joblib`
files trained for the ensemble-averaging experiment, now lazy-loaded and
cached (`_get_ensemble_models`, mirroring the existing `_get_model` pattern)
directly inside the live agent tool rather than a one-off analysis script.

Crucially, `uncertainty.mean` is **not** used anywhere as the reported
`probability` — the top-level `probability` field is still, exactly as
before, the single production model's own `predict_proba`. `uncertainty` is
strictly additive context. This sidesteps both failure modes found in the two
rejected fix attempts: there is no threshold to re-derive, no existing
verified number changes, and no decision-quality regression is possible,
because no decision-relevant number moved.

Live example (`forecast_tool("Jizan", "2025-08-23", "flash_flood")`):

```json
"probability": 0.1253,
"uncertainty": {"mean": 0.1143, "std": 0.0486, "range": [0.0433, 0.185], "n_members": 5}
```

The 5 models disagree by roughly ±0.05 around a ~0.11–0.13 probability here —
a real, moderate-confidence case, not a false-precision single number.

## Why this is a genuinely distinct signal from the two existing caveat fields

`forecast_tool` already returns two other trust signals, and the agent's
system prompt (`agent/03_agent.py`) is explicit that `uncertainty` is a third,
independent axis, not a restatement of either:

- **`reflexive_check`**: compares the ML model's probability against a
  *separate rule-based physical detection engine* on the same day's raw
  indicators — disagreement here means "the model and the physics-based rules
  see this differently."
- **`meteorological_metrics`** (POD/FAR/CSI/HSS): describes the model's
  *historical, threshold-dependent track record* — a fixed number per hazard,
  independent of the specific city/date being queried.
- **`uncertainty`** (new): measures *how much 5 independently-trained
  copies of the same model architecture disagree with each other on this
  exact input* — a per-query signal that can be large even when
  `reflexive_check` shows agreement and `meteorological_metrics` looks
  strong, or vice versa. It is orthogonal by construction: it says nothing
  about accuracy or calibration, only about model-internal (dis)agreement.

## Testing

`agent/02_test_tools.py` (151 tests total, 8 new): confirms the field is
present with exactly the 4 documented keys, `n_members == 5`, `std >= 0`,
`range` is a valid `[lo, hi]` pair with `lo <= hi`, `mean` lies inside
`range`, the top-level `probability` is reported independently of
`uncertainty.mean` (never silently overwritten), the field is deterministic
across repeated calls to the same input, and a bypass check that the 5 raw
ensemble `.joblib` files load successfully as 5 distinct model objects.

`FULL_SYSTEM_AUDIT.py` Section O (new, 9 checks): a full independent
bypass — reloads all 15 raw ensemble files straight from disk (not through
`tools.py`'s internal cache), re-derives each flash_flood seed's ROC-AUC from
scratch against an independently-rebuilt held-out test set and checks it
against `manifest.json` exactly; confirms the 5 members are genuinely
distinct model objects, not duplicated files; reconstructs the Jizan
2025-08-23 feature row from raw dataset arrays using the same public
building blocks `tools.py` itself uses (independent of `forecast_tool`'s own
internal call path) and recomputes `uncertainty.mean/std/range` from scratch,
matching the live tool's output exactly; asserts `uncertainty.mean` is
structurally distinct from the top-level `probability` (not a silent copy);
and confirms all 15 ensemble files (3 hazards x 5 seeds) are present on disk.

## Performance impact

None on the existing forecast path's `probability`, `reflexive_check`, or
`meteorological_metrics` values — all unchanged, byte-for-byte, from before
this feature. The only added cost is loading 5 small model files (~0.5-0.6MB
each) once per hazard (lazy, cached after first use) and running
`predict_proba` 5 extra times per `forecast_tool` call.

## Relationship to the two rejected fix attempts

| Attempt | Changes the point estimate? | Decision-quality (POD/CSI/HSS) impact | Deployed? |
|---|---|---|---|
| Isotonic recalibration (`CALIBRATION_REPORT.md`) | Yes | Significant regression | No |
| Ensemble averaging (`ENSEMBLE_REPORT.md`) | Yes (small) | Small regression, all 3 hazards | No |
| **Ensemble uncertainty (this feature)** | **No** | **None possible — no decision number moves** | **Yes** |

This is the direct payoff of treating "the model is honestly overconfident"
and "the alert thresholds must not regress" as two separate constraints
instead of one problem to solve with one number: expose the honest signal
*next to* the unchanged decision-relevant number, instead of trying to make
one number do both jobs.
