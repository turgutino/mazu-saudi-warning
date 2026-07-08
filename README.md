# MAZU · Saudi Arabia — Multi-Hazard Extreme Weather Early Warning

沙特多灾种极端天气早期预警系统 — MAZU (Multi-hazard · Alert · Zero-gap · Universal).

An early-warning system for **flash floods** and **heatwaves** over Saudi Arabia, built on
CMA 10 km meteorological indicators. It combines a **data-grounded causal knowledge graph**,
a **weighted spatial detection engine**, and (in progress) a **spatiotemporal forecast model**
and an **explainable warning agent**.

🌐 **Live site:** https://turgutino.github.io/mazu-saudi-warning/

---

## What it does

| Layer | Status | Description |
|-------|--------|-------------|
| Indicator pipeline | ✅ | Consolidates 365 daily NetCDF files (10 km grid) into one analysis-ready dataset of 20 core reliable indicators. |
| Knowledge graph | ✅ | 51 nodes / 170 edges — indicators, hazards, mechanisms (Red Sea Trough, moisture transport, orographic lifting), regions and **real 2025 events with observed values**. Interactive. |
| Detection engine | ✅ | Weighted multi-condition rules + spatial connected-component clustering. Validated against known 2025 events and a spatial-climatology check. |
| Forecast (ST model) | 🔜 | Predict tomorrow's extreme risk — moving beyond detection to true early warning. |
| Explainable agent | 🔜 | Natural-language warnings tracing each alert to its drivers and mechanism. |

---

## Highlights

- **Data-grounded, not assumed.** Every event node carries its real observed indicator values
  (e.g. 23 Aug Jizan extreme rain 254.9 mm; IVT 728 kg/m/s; Tmax 53.7 °C).
- **Physics recovered from data.** Data-driven correlations reproduce known meteorology —
  sea-surface temperature ↔ convective instability (r ≈ 0.68), moisture ↔ transport (r ≈ 0.74).
- **Spatially verified.** Annual hotspots emerge correctly: flash floods over the SW Asir
  mountains / Red Sea coast; heatwaves over the low, hot interior.

---

## Repository layout

```
index.html                 landing page (GitHub Pages)
kg_view.html               interactive knowledge graph
img/                       risk maps
pipeline/build_dataset.py  365 daily files -> consolidated dataset
kg/                        knowledge-graph builder + dashboard generator
model/                     detection engine + risk-map visualisation
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
