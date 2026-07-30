# Server setup (RTX 5090, AutoDL) — NOT run yet, needs explicit go-ahead per step

These commands are written but NOT executed. Run them on the rented server only after
confirming with the user at each numbered checkpoint below.

## Checkpoint A — environment
```bash
python -V   # expect 3.11+ per earth2studio requirement
nvidia-smi  # confirm RTX 5090, CUDA 13.x driver, ~32GB VRAM free
pip install earth2studio
```

## Checkpoint B — GraphCast Small model weights
```bash
# confirm before running: downloads GraphCast Small checkpoint (size not yet confirmed —
# check `du -sh` after download and report back before proceeding)
python -c "from earth2studio.models.px import GraphCastSmall; m = GraphCastSmall.load_default_package()"
```

## Checkpoint C — dry run (single day, before committing to all 365)
Run `02_run_graphcast_inference.py --date 2025-08-23 --dry-run` first (one known validated
event date, Jizan flash flood) and manually inspect the output NetCDF:
- Confirm the 5 expected variables are present (t2m, u10, v10, msl, total_precipitation)
- Confirm the cropped region matches [16-32N, 34-56E]
- Confirm values are physically plausible (e.g., t2m in a sane Kelvin/Celsius range for
  Saudi Arabia in August, not all-zero or all-NaN)

**This checkpoint is the first real test of whether `GraphCastSmall` + `GFS()` actually work
together — do not proceed to the full run until this is manually confirmed.**

## Checkpoint D — full run
Only after Checkpoint C output looks sane: run `02_run_graphcast_inference.py --start
2025-01-01 --end 2025-12-31`. Expect this to take a while (real duration unknown until timed
in Checkpoint C) — run with logging so partial progress is never lost, and make the script
resumable (skip days that already have an output file) in case it needs to be restarted.

**Each checkpoint requires the user's explicit "go ahead" before executing — do not chain
these automatically.**
