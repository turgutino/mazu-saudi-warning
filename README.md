# MAZU · Saudi Arabia — Multi-Hazard Extreme Weather Early Warning

沙特多灾种极端天气早期预警系统 — MAZU (Multi-hazard · Alert · Zero-gap · Universal).

An early-warning system for **flash floods**, **heatwaves**, and **dust storms** over Saudi Arabia, built on
CMA 10 km meteorological indicators. It combines a **literature-grounded causal knowledge
graph**, a **weighted spatial detection engine**, a **verified one-day-ahead forecast model**,
and an **explainable warning agent** (DeepSeek function calling).

🌐 **Live site:** https://turgutino.github.io/mazu-saudi-warning/

---

## Full system audit

`FULL_SYSTEM_AUDIT.py` independently traces 100 numbers — from the consolidated
dataset, the knowledge graph's event values, the causal citations, an
agent tool's output, and eight post-Layer-4 extensions (terrain elevation,
population context, an A/B ablation test re-derived from raw saved
transcripts, a reflexive self-check whose two headline findings are
independently re-derived a third time directly from the raw source files,
a 5th agent tool closing a self-found KG utilization gap, CAP 1.2 alert
generation independently re-parsed from the actual XML output, WMO-
standard POD/FAR/CSI/HSS verification metrics recomputed from a fresh
confusion matrix against the saved production models, and a reliability-
diagram calibration check recomputed from scratch against the saved models)
— back to the raw 5GB source data (365 daily NetCDF files), plus checks the
deployed GitHub site matches the local repo exactly.
**Result: 100/100 passed, zero fabricated values found.** Full log in
[`AUDIT_RESULTS.txt`](AUDIT_RESULTS.txt).

---

## What it does

| Layer | Status | Description |
|-------|--------|-------------|
| Indicator pipeline | ✅ | Consolidates 365 daily NetCDF files (10 km grid) into one analysis-ready dataset of 22 core reliable indicators. |
| Knowledge graph | ✅ | 60 nodes / 183 edges — indicators, hazards (now 3: flash flood, heatwave, **dust storm**), mechanisms, regions, **real 2025 events with observed values**, and **6 peer-reviewed citations** grounding 4 of 5 mechanisms in verbatim-verified literature text. Interactive. |
| Detection engine | ✅ | Weighted multi-condition rules + spatial connected-component clustering. Validated against known 2025 events and a spatial-climatology check. |
| Forecast (t→t+1) | ✅ | Gradient-boosted spatiotemporal model. Heatwave ROC-AUC 0.971 (PR-AUC 0.795); flash-flood ROC-AUC 0.873 — plus WMO-standard POD/FAR/CSI/HSS at each hazard's operational threshold (see below). A GNN variant was also tested and honestly reported (mixed result, not deployed). |
| Explainable agent | ✅ | DeepSeek function-calling agent wiring 6 tools — forecast, causal KG, live conditions, similar events, region risk, and CAP 1.2 alert generation — into grounded answers. 143 tool tests + 4 end-to-end scenarios, all passing. See `agent/LAYER4_REPORT.md` and the [worked examples](agent_view.html). |

---

## Extensions: terrain context, impact framing, and a live A/B ablation test

After reviewing other teams' project reports and a published multi-agent
early-warning system (MAESTRO, *npj Artificial Intelligence*, 2026,
Zhejiang University), three targeted extensions were added and independently
tested — see `agent/EXTENSIONS_REPORT.md` for full methodology and results:

- **Terrain/elevation context.** `forecast_tool` now flags mountain-city
  forecasts (Abha, Taif, ≥1500 m) as lower-confidence. Building this
  surfaced a genuine, previously-undocumented limitation of the existing
  model: in the steep Asir range, the model's coarse feature grid can land
  on a cell ~750 m lower than a city's true elevation — exactly the case
  this feature exists to catch.
- **A/B ablation test (live, 8 real DeepSeek calls).** The same 4 "why"
  questions were run through the agent with vs. without `causal_kg_tool`.
  With it: 4/4 answers cited a real driving mechanism and literature.
  Without it: 0/4 did — and, critically, 0/4 hallucinated a mechanism
  anyway; the agent correctly said it had no grounded explanation. Full
  transcripts in `agent/ABLATION_REPORT.md`.
- **Impact-based context.** `forecast_tool` now returns each city's
  population (Saudi Census 2022, GASTAT), explicitly labelled as reference
  context only — not an exposure estimate — per WMO's impact-based warning
  guidance.
- **Reflexive self-check.** `forecast_tool` now cross-checks the ML
  probability against Layer 1's independent rule-based detection engine on
  the same day's indicators (a Reflexion-style consistency check, per
  MAESTRO). This surfaced two genuine, independently-verified findings: on
  23 Aug (Jizan), the physical preconditions for flash flooding (CAPE, IVT,
  PWAT) were already elevated the day before, while the model gave only a
  12.5% probability — a real precursor signal the model underweighted; on
  25 Jul (Mecca), the model correctly flagged an anomalous heatwave (+3.68°C
  above local climatology) that the detection engine's fixed absolute
  thresholds missed entirely — real evidence the ML model adds value beyond
  simple thresholding. See `agent/REFLEXIVE_CHECK_REPORT.md`.
- **Similar historical events (4th agent tool).** `similar_events_tool`
  compares a city/date's indicators against the KG's 5 known real events via
  z-scored similarity, making those event nodes agent-usable rather than
  only browsable. Found a real, investigated quirk: an event's coordinates
  are its own grid-cell maximum (the storm centroid), often tens of km from
  a same-named city's center, so a same-city/same-day query can legitimately
  score LOW similarity to "its own" event — now surfaced explicitly via an
  `event_distance_from_city_km` field rather than silently confusing.
  Verified with a real found-and-independently-checked positive match too
  (Mecca 08-04 vs. the 07-25 Empty Quarter heat event, 46.1%, despite being
  1,659km apart — both were anomalously hot for their own conditions). See
  `agent/SIMILAR_EVENTS_REPORT.md`.
- **Dust storm — a 3rd hazard, full depth.** Not detection-only: 2 new raw
  indicators added to the dataset (wind10_speed, dewpoint_depression_c,
  processed from all 365 raw files), a data-grounded detection rule, a
  genuinely trained and out-of-sample-validated forecast model (ROC-AUC
  0.8866, PR-AUC 0.1635 — comparable to flash-flood), and real KG grounding
  that required **zero new literature extraction**: the KG's existing Shamal
  citation (Yu et al. 2016) already states *"The summer Shamal is the major
  driver of dust storm activity across the Arabian Peninsula"* — dust_storm
  was correctly re-pointed at that pre-existing grounding rather than
  inventing a new one. A real methodological catch during testing: the first
  known-event validation attempt accidentally used a training-period date;
  caught and corrected to a genuine out-of-sample test-period event before
  being reported. See `agent/DUST_STORM_REPORT.md`.
- **region_risk_tool (5th agent tool) — closing a self-found KG utilization
  gap.** An honest audit of edge-type usage found only 3 of 11 KG edge types
  (~20% of edges) were ever queried by any tool; `at_risk_of`/`exposed_to`
  (region-hazard exposure) existed in the graph but nothing read them. This
  city-first tool ("what should I worry about in Jizan", vs. the other 4
  tools' hazard-first "why does flash_flood happen") closes part of that gap
  (~40% of edges now used) and surfaced two real findings along the way: not
  every KG hazard has a trained model (`coastal` — handled gracefully, no
  fabricated forecast), and a genuine pre-existing KG data-quality gap
  (Jeddah is at_risk_of heatwave but its exposed_to mechanisms don't overlap
  with heatwave's actual drivers — traced to the original hand-encoded
  domain knowledge, disclosed rather than silently patched). See
  `agent/REGION_RISK_REPORT.md`.
- **cap_alert_tool (6th agent tool) — CAP 1.2 standards-compliant alerts.**
  Converts a forecast into a real CAP 1.2 (Common Alerting Protocol, the
  OASIS standard MAZU's own national framework is built on) XML alert,
  ready to plug into broadcast/siren/SMS infrastructure — closing the gap
  between "scientifically correct" and "operationally integrable". Severity
  is derived from `DetectionEngine`'s own thresholds (not a second invented
  scale), certainty from the reflexive check's model/rule-engine agreement,
  and `status` is hardcoded to `Exercise` (never `Actual`) since this is a
  historical-dataset demo, not a live feed. A real, investigated finding:
  a genuine `consistent_elevated` day can still fall below the (deliberately
  higher) CAP alert threshold — the reflexive check and the alert-issuance
  bar answer different questions on purpose. See `agent/CAP_REPORT.md`.
- **WMO-standard verification metrics (POD/FAR/CSI/HSS).** Adopted after
  reviewing another team's evaluation methodology. Computed from the
  already-saved production models (no retraining) at each hazard's own
  operational threshold — the same value CAP severity already uses.
  Surfaces a real, disclosed finding threshold-independent ROC-AUC alone
  cannot: flash_flood has a strong ROC-AUC (0.873) yet a genuinely low POD
  (0.10) at its fixed threshold, because true flash-flood days are an
  extremely rare event (~0.5% of test-set cells) — reported honestly rather
  than only showing the more flattering number. See
  `agent/METEOROLOGICAL_METRICS_REPORT.md`.
- **Reliability diagrams — is the probability itself trustworthy?** ROC-AUC
  and POD/FAR/CSI/HSS answer different questions than "when the model says
  70%, does it happen ~70% of the time?" Binning every test-set prediction
  by probability found that all 3 hazards are **systematically
  overconfident** at the high end (flash flood's "95.7%" bucket only sees
  the event 54.3% of the time) — and that flash flood's low aggregate ECE
  in isolation is itself misleading, since ~97% of samples sit in one
  near-zero bucket that masks the same pattern. A follow-up experiment
  (`model/10_calibration_fix.py`, leak-free 3-way chronological split)
  confirmed isotonic recalibration genuinely improves Brier score for all 3
  hazards, but was kept as a tested, documented finding rather than
  deployed — swapping in calibrated probabilities would shift ~150
  already-verified test numbers and CAP's own severity thresholds. See
  `agent/CALIBRATION_REPORT.md`.

All 143 unit tests pass (`agent/02_test_tools.py`, up from 32); 50 further
independent checks verify the calibration analysis (`model/09b_test_calibration.py`,
`model/10b_test_calibration_fix.py`).

---

## Highlights

- **Data-grounded, not assumed.** Every event node carries its real observed indicator values
  (e.g. 23 Aug Jizan extreme rain 254.9 mm; IVT 728 kg/m/s; Tmax 53.7 °C).
- **Literature-grounded causality, not just my own assertions.** 21 candidate causal triples
  were extracted from 7 real, cited publications (e.g. de Vries et al. 2013, JGR-Atmospheres;
  Yu et al. 2016, JGR-Atmospheres) using an LLM with a mandatory verbatim-quote field; every
  quote is automatically re-checked against the source text, and one triple that passed the
  automatic check but was circular on manual review was disclosed and excluded — see
  `kg/causal/LAYER3_REPORT.md` for the full audit trail.
- **Physics recovered from data.** Data-driven correlations reproduce known meteorology —
  sea-surface temperature ↔ convective instability (r ≈ 0.68), moisture ↔ transport (r ≈ 0.74).
- **Spatially verified.** Annual hotspots emerge correctly: flash floods over the SW Asir
  mountains / Red Sea coast; heatwaves over the low, hot interior.
- **Genuine forecasting, not same-day detection.** The model predicts tomorrow's risk from
  today's indicators (trained on Jan–Jun 2025, tested on unseen Jul–Dec), verified against
  known events and negative controls — see `model/forecast_report.txt`.
- **The knowledge graph is functional, not decorative.** The agent programmatically queries
  it on every question (the same way a production detection system would), rather than it
  being a browsable diagram nobody's code actually reads.

---

## Repository layout

```
index.html                 landing page (GitHub Pages)
kg_view.html               interactive knowledge graph
agent_view.html            real, verified agent transcripts
img/                       risk maps + forecast visualisations
pipeline/build_dataset.py  365 daily files -> consolidated dataset
kg/                        knowledge-graph builder + dashboard generator
kg/causal/                 literature-grounded causal extraction (DeepSeek + CoT)
agent/                     forecast + causal-KG + conditions tools, DeepSeek agent
model/                     detection engine, forecast models, risk-map visualisation
```

## Region

| Item | Value |
|------|-------|
| Latitude | 16 N – 32 N |
| Longitude | 34 E – 56 E |
| Resolution | 0.1° (~10 km), 160 × 220 grid |
| Period | 2025-01-01 – 2025-12-31 (365 days) |

## Data

The 5 GB indicator NetCDF files (`saudi_indicators_YYYYMMDD.nc`) are **not** included in the
repository. Place them locally and point the pipeline at them.
