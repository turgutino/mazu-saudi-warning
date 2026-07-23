# Reproducing the verification report's numbers

This reproduces every real-vs-predicted risk score cited in
`MAZU_Real_Hadise_Dogrulama_EN.html` (12-event verification report),
independently of that report's text — it recomputes the numbers from the
underlying model and rule engine and compares them.

## What it checks

`reproduce_verification.py` calls `real_grid()` / `predicted_grid()`
(`build_grid.py`) for every (date, hazard, location) the report cites a
number for, and asserts the recomputed value matches within ±0.02.

- `real_grid()` reads `DetectionEngine.risk_field()` — the independent,
  rule-based ground truth.
- `predicted_grid()` reproduces `forecast_tool`'s feature construction
  vectorized across the full grid. Its own `__main__` block cross-checks one
  cell against `forecast_tool()` directly and asserts an exact match before
  anything else is trusted.

48 individual claims across 28 point rows + 4 multi-day range rows, covering
all 12 reviewer-provided calendar events. 4 items are explicitly out of
scope (raw meteorological values — wind speed, temperature, precipitation —
are not risk scores and aren't produced by these two functions); they're
listed by name in the script's output, not silently skipped.

## Requirements

- Conda/venv with the project's `ml` environment (numpy, xarray, the trained
  model artifacts under `mazu-system/agent`, and `mazu-system/data/mazu_dataset.nc`
  present).
- Run from `Competation/scratch_map/` so `build_grid.py`'s relative imports
  resolve.

## Run it

```bash
cd Competation/scratch_map
python reproduce_verification.py
```

Exit code 0 + "PASS" at the end means every claim matched. Exit code 1 lists
each mismatch by name — regenerate the maps/report text for that event
before trusting it again.

## Last verified

2026-07-23: 48/48 claims passed, 0 mismatches, against
`MAZU-FENGYUN/reports/MAZU_Real_Hadise_Dogrulama_EN.html`.
