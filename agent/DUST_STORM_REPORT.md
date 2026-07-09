# MAZU Extension — Dust Storm: A 3rd Hazard, Full Depth

## What this is

The system now covers **3 hazards** (flash_flood, heatwave, dust_storm), not
2. Dust storm was added with the SAME depth as the original two: new raw
indicators pulled into the dataset, a data-grounded detection rule, a
genuinely trained and validated t→t+1 forecast model, real knowledge-graph
grounding (reusing an existing literature citation), and full integration
into all 4 agent tools — not a shallow "detection only" addition.

## 1. New raw indicators added to the dataset

`wind10_speed` and `dewpoint_depression_c` exist in the raw 5GB source but
were not in the original 16-variable consolidated dataset. Both are
time-varying (unlike the static `orography` added earlier), so all 365 raw
files were read (`pipeline/08_add_dust_indicators.py`, ~27s using the fast
netCDF4 selective-read pattern from `01_build_dataset.py`, not xarray's slow
full-file parse). File-date order was explicitly asserted to match the
consolidated dataset's time axis before merging (not assumed).

## 2. Detection rule (Layer 1)

Added to `model/01_detection_engine.py`'s `RULES` dict, following the exact
same convention as flash_flood/heatwave: percentile-derived thresholds
(~p93-p95), a primary condition (surface wind, the direct lifting
mechanism) plus 3 supporting conditions (low-level jet, dryness, VPD).
**Both wind AND dryness are required** — pure high wind at a humid coastal
cell (found during exploration: 2025-07-26, Arabian Sea, wind10_speed=20.7
m/s but dewpoint_depression_c only ~3°C) does not produce dust; that
combination was explicitly checked and correctly does NOT fire the dust rule.

**Known event found and validated**: searching across all cities/dates/grid
cells for the day with the most cells jointly satisfying all 4 conditions
found **2025-06-19** — 6,380 cells, peak at (31.9°N, 44.3°E), which
independently falls within the KG's already-cited Shamal season (May-August,
Yu et al. 2016). Negative controls (2025-01-15, 2025-02-10) show 0 detected
clusters.

## 3. Forecast model (t→t+1)

No pre-computed `dust_storm` label existed in the dataset (unlike
`flash_flood_risk`/`heatwave_day_flag`, which were teacher-provided). Rather
than inventing a new label definition, the label was built directly from
Layer 1's own already-validated rule (`risk_threshold=0.55`) — reusing
existing, tested domain knowledge instead of adding an unverified second
definition of what "dust storm" means.

Same methodology as the other 2 hazards: `HistGradientBoostingClassifier`,
Jan-Jun train / Jul-Dec test (no leakage), rare-event metrics.

**Result: ROC-AUC=0.8866, PR-AUC=0.1635** — comparable to (slightly better
than) flash_flood (0.873/0.089), for a similarly rare event (~2% positive
rate). Full report: `model/dust_storm_forecast_report.txt`.

**A real methodological catch during testing**: the first known-event
validation attempt used 2025-06-19 (the detection-validation event) to check
forecast skill — but 06-19 falls in the TRAINING period (≤06-30), so this
would have validated against data the model already saw, not genuine
forecast skill. Caught before being reported: re-searched restricted to the
Jul-Dec test window and found 2025-07-06 instead (widespread, largest
dust-rule cluster count in the test period). On that genuinely unseen day:
mean probability 0.212 (vs. ~2% base rate), 1,715/8,800 cells with p>0.5,
max p=0.992 — real, out-of-sample forecast skill, not memorization.

## 4. Knowledge graph integration

The KG **already had** a "dust" hazard node (from the original structural
build) with `thermal_low` as its mechanism — and `thermal_low` was **already
literature-grounded** via the existing Yu et al. (2016) Shamal citation,
whose own evidence quotes literally state *"The summer Shamal is the major
driver of dust storm activity across the Arabian Peninsula"* and *"capable
of lifting dust and transporting it to the Persian Gulf and the Arabian
Peninsula."* Grounding dust_storm required **zero new literature
extraction** — only renaming the hazard id from `dust` to `dust_storm` (for
consistency with the model/agent naming) and adding 2 new Indicator nodes
(`wind10_speed`, `dewpoint_depression_c`) plus their `contributes_to` edges.

A 6th Event node was added following the exact same auto-detection
methodology as the original 5 (`kg/01_build_structural_kg.py`'s
`EVENT_DEFS` pattern: annual grid-max of a headline variable): wind10_speed's
true 2025 maximum, 2025-07-26, 20.7 m/s, near the Arabian Sea.

**Full KG integrity re-verified after the rename**: 0 dangling edges (every
edge's source/target resolves to a real node), 0 duplicate node ids, the old
`dust` id fully removed (not left as an orphaned duplicate). KG grew from
57/176 to **60/183** nodes/edges. The source generator script
(`kg/01_build_structural_kg.py`) was also updated for future consistency
(not re-run, to avoid overwriting the hand-merged Layer 3 citations already
in `kg_data.json`).

## 5. Agent tools (all 4 updated)

- `forecast_tool`: dust_storm feature vector matches the model's exact
  training order (17 base vars + wind10_speed + dewpoint_depression_c +
  lat/lon/day_of_year); reports its own distinct ROC-AUC.
- `causal_kg_tool("dust_storm")`: returns thermal_low, literature-grounded,
  with the Shamal citation.
- `similar_events_tool`: compares against the 1 dust_storm event in the KG.
- `reflexive_check`: works generically (no dust-specific code needed, since
  it reads `DETECTION_RULES[hazard]` dynamically) — found and independently
  verified a real `consistent_elevated` case (Dammam, 2025-07-06, both
  signals agreeing: model 90.3%, detection rule 0.60, driven by
  wind10_speed=9.0 and wind850_speed=18.4 at the raw source).

## Testing

17 new checks in `02_test_tools.py` (90/90 total, up from 73), covering all
4 tools with dust_storm, a found-and-independently-verified elevated case
(Dammam), a negative control, and error handling. `FULL_SYSTEM_AUDIT.py`
Section J re-derives the new dataset variables, the event's headline value,
KG integrity, and the reflexive-check score all directly from the raw 5GB
source (78 total checks; 76/78 passed locally — the 2 "failures" are
Section F's expected remote-vs-local drift check, which will pass once this
work is pushed to GitHub).

## Live agent verification

Asked live: *"What is the dust storm risk in Dammam for 2025-07-06, and what
physically causes dust storms in Saudi Arabia?"* — the agent correctly
reported 90.3% probability, noted the reflexive-check agreement, cited the
Yu et al. (2016) Shamal mechanism with a verbatim quote, and gave a sensible
recommendation — end-to-end, no fabrication.
