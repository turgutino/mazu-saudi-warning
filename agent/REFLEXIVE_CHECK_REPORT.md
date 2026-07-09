# MAZU Extension — Reflexive Self-Check (Model vs. Independent Rule-Based Detection)

## What this is

`forecast_tool` now returns a `reflexive_check` field that cross-validates the
trained ML model's probability against Layer 1's **independent, already-tested**
rule-based detection engine (`model/01_detection_engine.py`), evaluated on the
SAME day's raw indicators the model used as input. Two independently-built
signals — a trained classifier and hand-set, data-grounded physical thresholds
(percentile-derived, verified in that file's own test suite months earlier) —
should broadly agree; when they don't, that disagreement is surfaced to the
user rather than hidden.

**Source of the idea:** the MAESTRO paper (Wang et al., *npj Artificial
Intelligence*, 2026) validates agent outputs against independent domain
constraints using a Reflexion-style self-check (Shinn et al., 2023). We
re-derived the same principle at our own scale using a resource we already
had — Layer 1's detection engine — rather than building new infrastructure.

## Design

- Threshold: both signals (model probability, detection risk score) are
  compared against a fixed, documented cutoff of **0.3** — chosen as a round
  value clearly above typical rare-event base rates (0.5-5%) but below the
  detection engine's own event-cluster threshold (0.5-0.55), so it flags
  meaningful signal without requiring a fully clustered event. Not tuned to
  produce any particular result.
- Four consistency labels: `consistent_low`, `consistent_elevated`,
  `model_higher_than_detection`, `detection_higher_than_model`.
- Elevation context (Extension 1) taught us to use full-resolution grid
  lookups rather than the model's coarser stride-2 feature grid where
  possible; the same discipline applies here — the detection engine runs on
  the full grid.

## Testing (matches the session's standing methodology: verify in isolation,
## investigate every anomaly as a possible real finding before assuming the
## test is wrong)

11 new checks added to `02_test_tools.py` (55/55 total, up from 45). Three
notable results, EACH independently re-verified against raw indicator values
(not just re-reading this tool's own output) before being written into the
test suite as expected behavior:

### Finding 1 — Riyadh calm day (2025-11-06): `consistent_low` ✅
Both signals agree there's nothing going on. Expected, uninteresting, correct.

### Finding 2 — Jizan flash-flood, 2025-08-23: `detection_higher_than_model`
The forecast uses 2025-08-22's indicators (the day BEFORE the actual 254.9mm
storm). Independently verified directly from the raw source file:
`daily_precip_total=0.5mm` (rain had NOT started — correctly below the
detection engine's 10mm anchor condition), but `flash_flood_risk=2.0`,
`cape=2292 J/kg`, `ivt=227.7 kg/m/s`, `pwat=61.1 kg/m2` — all four supporting
conditions above their data-grounded thresholds. Detection score:
0.20+0.15+0.15+0.10 = **0.60**. The trained model's probability for that same
transition: **12.5%**.

**Interpretation (disclosed, not hidden):** the atmosphere was already
physically "primed" for a flash flood the day before it happened, but the ML
model substantially underweighted that precursor signature. This is
consistent with — and adds independent evidence for — a limitation already
disclosed in `model/forecast_report.txt` (flash-flood forecasting is
"correctly located but under-confident" for this exact event). This is a
genuine finding about a real, pre-existing model limitation, not a bug
introduced by this extension.

### Finding 3 — Mecca heatwave, 2025-07-25: `model_higher_than_detection`
Precursor day 2025-07-24, independently verified from raw source:
`tmax_c=42.98°C` (below the detection engine's absolute 45°C threshold) and
`heat_index_c=37.0°C` (below its 40°C threshold) — detection score **0.0**,
nothing fires. But `tmax_anomaly_c=+3.68°C` above Mecca's own climatology,
and the ML model (trained on the anomaly-aware `heatwave_day_flag` label:
`tmax_c >= max(40, climatology+5)`) correctly assigns **62.8%** probability.

**Interpretation (disclosed, not hidden):** this is NOT a model error — it is
the detection engine's absolute-threshold design missing a genuinely
anomalous (climatologically hot for Mecca specifically) day that the
anomaly-aware ML model correctly catches. This is a real demonstration that
the ML model adds value beyond fixed-threshold detection, and matches an
anomaly-vs-absolute-threshold distinction already investigated and documented
in `02_test_tools.py`'s Mecca/Abha heatwave-ranking test from Layer 4.

## Verification trail

- `agent/tools.py` — `_reflexive_check()`, reuses `model/01_detection_engine.py`'s
  already-tested `DetectionEngine`/`RULES` rather than inventing new thresholds.
- `agent/02_test_tools.py` — 11 new checks, including a bypass test that calls
  `DetectionEngine` directly (independent of `forecast_tool`) to confirm no drift.
- `FULL_SYSTEM_AUDIT.py` Section H — both findings re-derived a THIRD time,
  directly from the raw 5GB source files (bypassing even the consolidated
  dataset), confirming the numbers all the way back to source.
- Live agent sanity check: asking about the Jizan case produced an answer that
  correctly surfaced the caveat ("⚠️ Important Caveat — Detection vs. Model
  Mismatch... Caution is warranted") rather than silently reporting only the
  12.5% figure.
