# MAZU · Saudi Arabia — Multi-Hazard Extreme Weather Early Warning

沙特多灾种极端天气早期预警系统 — MAZU (Multi-hazard · Alert · Zero-gap · Universal).

An early-warning system for **flash floods** and **heatwaves** over Saudi Arabia, built on
CMA 10 km meteorological indicators. It combines a **literature-grounded causal knowledge
graph**, a **weighted spatial detection engine**, a **verified one-day-ahead forecast model**,
and (in progress) an **explainable warning agent**.

🌐 **Live site:** https://turgutino.github.io/mazu-saudi-warning/

---

## What it does

| Layer | Status | Description |
|-------|--------|-------------|
| Indicator pipeline | ✅ | Consolidates 365 daily NetCDF files (10 km grid) into one analysis-ready dataset of 20 core reliable indicators. |
| Knowledge graph | ✅ | 57 nodes / 176 edges — indicators, hazards, mechanisms, regions, **real 2025 events with observed values**, and **6 peer-reviewed citations** grounding 4 of 5 mechanisms in verbatim-verified literature text. Interactive. |
| Detection engine | ✅ | Weighted multi-condition rules + spatial connected-component clustering. Validated against known 2025 events and a spatial-climatology check. |
| Forecast (t→t+1) | ✅ | Gradient-boosted spatiotemporal model. Heatwave ROC-AUC 0.971 (PR-AUC 0.795); flash-flood ROC-AUC 0.873. A GNN variant was also tested and honestly reported (mixed result, not deployed). |
| Explainable agent | ✅ | DeepSeek function-calling agent wiring the forecast model, causal KG and live conditions into grounded answers. 27 tool tests + 4 end-to-end scenarios, all passing. See `agent/LAYER4_REPORT.md` and the [worked examples](agent_view.html). |

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
