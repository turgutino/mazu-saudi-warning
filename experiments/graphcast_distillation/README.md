# Experiment: GraphCast (Small, 1°) distillation feature

**Status: experimental, on branch `experiment/pangu-distillation`, NOT merged to main.**

## Goal
Test whether adding GraphCast's own global forecast (for the same day/region) as an extra
input feature improves the 3 hazard models — specifically flash_flood's POD, which is the
one honestly-disclosed weak point in the current system (§11 of the report).

## Model choice history (kept for transparency, not just the final answer)
1. First considered Pangu-Weather (39-year ERA5 training, documented as more robust than IFS,
   real heatwave case-study performance). Rejected after checking its exact output variable
   list against the original repo: Pangu outputs MSLP/U10/V10/T2M + Z/Q/T/U/V at 13 pressure
   levels — **no precipitation variable**, which makes it weak specifically for flash_flood,
   our main target.
2. Switched to **GraphCast**, confirmed via search to output total precipitation (TP) as one
   of its 5 surface variables, alongside T2M/MSLP/U10/V10. Also has the strongest published
   accuracy record found (beat ECMWF HRES on 90% of 1,380 verification targets, Science 2023).
3. Full-resolution GraphCast Operational (0.25°) needs ~60GB VRAM — more than our rented
   RTX 5090 (32GB). Using **GraphCast Small (1°)** instead, which needs ~16GB and fits
   comfortably. This is a real resolution trade-off, disclosed here, not hidden.

## License
Model weights: CC-BY-NC-SA 4.0 (non-commercial) — compatible with this academic competition
project. Code: Apache 2.0.

## Safety constraints (per user instruction — must stay fully reversible)
- All work stays on this branch and in this folder until an explicit, measured improvement
  is confirmed and the user approves a merge.
- `agent/tools.py`, `model/saved_models/*.joblib`, and all files outside `experiments/` are
  NOT touched until step 4 passes.
- Existing 237 unit tests / 121 audit checks must still pass unmodified after any merge.

## Pipeline (4 steps)

1. **`01_server_setup.md`** — commands to run ON THE RENTED GPU SERVER (not run yet, needs
   server SSH access + explicit go-ahead for the download step).
2. **`02_run_graphcast_inference.py`** — runs ON THE SERVER. For each of the 365 days of 2025,
   runs GraphCast Small globally (GFS initial conditions, matching our own t-1→t methodology),
   crops to [16-32N, 34-56E] (same box as `mazu_dataset.nc`), saves the cropped fields
   (t2m, u10, v10, msl, total precipitation) as one small NetCDF per day.
3. **`03_integrate_feature.py`** — runs LOCALLY. Regrids the GraphCast output (1°, coarser
   than our 0.1° CMA grid) onto our existing grid (nearest-neighbor — no fabricated precision),
   adds the new columns as extra features, writes a NEW dataset file
   (`mazu_dataset_with_graphcast.nc`) — the original `mazu_dataset.nc` is never overwritten.
4. **`04_retrain_and_compare.py`** — runs LOCALLY. Retrains flash_flood (and heatwave,
   dust_storm) using the exact same methodology as `model/03_forecast_baseline.py` /
   `07_dust_storm_forecast.py` (same HistGradientBoostingClassifier params, same time-based
   train/test split, same POD/FAR/CSI/HSS metrics), once WITHOUT and once WITH the new
   feature(s), and prints a side-by-side comparison. Saves new model files under
   `experiments/graphcast_distillation/models_candidate/` — never overwrites
   `model/saved_models/*.joblib`.

## Decision rule
Deploy only if the new feature improves POD (or CSI/HSS) for the target hazard(s) **without**
degrading the others beyond a small, stated tolerance — same standard applied to the
isotonic-recalibration and ensemble experiments in `model/10_calibration_fix.py` /
`model/12_ensemble.py`, both of which failed this bar and were correctly not deployed.
If it fails the bar, document the honest negative result (consistent with project's
disclosure-first practice) and stay on the current, verified models.

## Known open risks (not yet resolved, will be checked at each step)
- Whether earth2studio's `GraphCastSmall` wrapper accepts `GFS()` as a data source has been
  found in docs for `GraphCastOperational`, not explicitly confirmed yet for `GraphCastSmall`
  — first real check happens at server Checkpoint C (single-day dry run).
- Real runtime for 365 sequential days is unknown until timed on the actual rented GPU.
- GFS historical archive may have gaps on specific dates; the inference script must handle
  missing days gracefully (skip + log, not crash).
