# =============================================================================
# GraphCast (Small, 1deg) inference for Saudi region distillation feature
#
# RUNS ON THE SERVER ONLY (needs GPU with ~16GB+ VRAM). NOT run yet from here.
#
# For each date in the requested range, forecasts date -> date+1 (matching our
# own forecast_tool's t-1 -> t methodology exactly: uses date's GFS conditions
# to forecast date+1), crops the global output to the Saudi region bounding box
# used throughout this project (16-32N, 34-56E, same as mazu_dataset.nc), and
# saves ONE small NetCDF per day so a crash/restart never loses prior progress.
#
# Usage (server only):
#   python 02_run_graphcast_inference.py --date 2025-08-23 --dry-run
#   python 02_run_graphcast_inference.py --start 2025-01-01 --end 2025-12-31
# =============================================================================
import argparse
import os
import shutil
import sys
import traceback
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

# GFS input cache grows ~144MB/day (measured 2026-07-19, 3-day sample) and is
# NOT needed once a day's output is written -- only the model weight cache
# (earth2studio/graphcast) is reused across days. Cleared after every day so
# a long multi-day run never exhausts a small container disk.
GFS_CACHE_DIR = os.path.expanduser("~/.cache/earth2studio/gfs")


def _clear_gfs_cache() -> None:
    if os.path.isdir(GFS_CACHE_DIR):
        shutil.rmtree(GFS_CACHE_DIR, ignore_errors=True)

# Saudi region bounding box -- must match mazu_dataset.nc exactly (see
# model/01_detection_engine.py / agent/tools.py CITIES for the same box).
LAT_MIN, LAT_MAX = 16.0, 32.0
LON_MIN, LON_MAX = 34.0, 56.0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_graphcast_daily")
os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED_VARS = ["t2m", "u10m", "v10m", "msl", "tp06"]  # tp06 = 6hr total precip (confirmed
# via a live GraphCastSmall run, 2026-07-19: real output var name, not "total_precipitation_6hr")


def run_one_day(date_str: str, dry_run: bool = False) -> str | None:
    """Forecast date_str -> date_str+1, crop to Saudi region, save one NetCDF.
    Returns the output path, or None if this day was skipped (already exists
    or GFS data unavailable)."""
    out_path = os.path.join(OUT_DIR, f"graphcast_{date_str}.nc")
    if os.path.exists(out_path):
        print(f"[SKIP] {date_str}: already exists at {out_path}")
        return None

    # Imports deferred so this file can be inspected/linted locally without
    # earth2studio installed.
    from earth2studio.data import GFS
    from earth2studio.io import NetCDF4Backend
    from earth2studio.models.px import GraphCastSmall
    from earth2studio.run import deterministic as run

    tmp_path = out_path + ".tmp_global.nc"

    try:
        package = GraphCastSmall.load_default_package()
        model = GraphCastSmall.load_model(package)
        data = GFS()
        io = NetCDF4Backend(file_name=tmp_path, backend_kwargs={"mode": "w"})

        # 1 step of the model's native lead time = date_str -> date_str+1,
        # mirroring forecast_tool's t-1 -> t design exactly.
        run([f"{date_str}T00:00:00"], 1, model, data, io)

        # Crop the global output down to just the Saudi region and keep only
        # the 5 variables we actually need -- no need to ship the full global
        # grid around.
        ds_global = xr.open_dataset(tmp_path)
        missing = [v for v in EXPECTED_VARS if v not in ds_global.variables]
        if missing:
            raise ValueError(
                f"{date_str}: expected variables missing from GraphCastSmall output: {missing}. "
                f"Available: {list(ds_global.variables)}"
            )

        ds_crop = ds_global[EXPECTED_VARS].sel(
            lat=slice(LAT_MAX, LAT_MIN),  # most global grids are north-to-south
            lon=slice(LON_MIN, LON_MAX),
        )

        if dry_run:
            print(f"[DRY RUN] {date_str}: cropped shape = "
                  f"{ {v: ds_crop[v].shape for v in EXPECTED_VARS} }")
            for v in EXPECTED_VARS:
                arr = ds_crop[v].values
                print(f"    {v}: min={np.nanmin(arr):.3f} max={np.nanmax(arr):.3f} "
                      f"nan_frac={np.isnan(arr).mean():.3f}")
            ds_global.close()
            os.remove(tmp_path)
            return None  # dry run never writes the final file

        ds_crop.to_netcdf(out_path)
        ds_global.close()
        os.remove(tmp_path)
        print(f"[OK] {date_str} -> {out_path}")
        return out_path

    except Exception as e:
        print(f"[FAIL] {date_str}: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None

    finally:
        # Runs on every exit path (success, dry-run, exception) so disk never
        # accumulates across a long multi-day run.
        _clear_gfs_cache()


def daterange(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    d = d0
    while d <= d1:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="single date YYYY-MM-DD (use with --dry-run for Checkpoint C)")
    ap.add_argument("--start", help="range start YYYY-MM-DD")
    ap.add_argument("--end", help="range end YYYY-MM-DD")
    ap.add_argument("--dates-file", help="path to a text file, one YYYY-MM-DD date per line "
                     "(for sparse/non-contiguous date sets, e.g. the 12-event calendar windows)")
    ap.add_argument("--dry-run", action="store_true",
                     help="inspect output only, do not save the final per-day file")
    args = ap.parse_args()

    if args.date:
        run_one_day(args.date, dry_run=args.dry_run)
    elif args.dates_file or (args.start and args.end):
        if args.dates_file:
            with open(args.dates_file) as f:
                dates = [line.strip() for line in f if line.strip()]
        else:
            dates = list(daterange(args.start, args.end))

        ok, failed, skipped = 0, [], 0
        for i, d in enumerate(dates):
            print(f"\n=== [{i+1}/{len(dates)}] {d} ===", flush=True)
            result = run_one_day(d, dry_run=args.dry_run)
            if result:
                ok += 1
            elif os.path.exists(os.path.join(OUT_DIR, f"graphcast_{d}.nc")):
                skipped += 1
            else:
                failed.append(d)
        print(f"\nDone. ok={ok} skipped(existing)={skipped} failed={len(failed)}")
        if failed:
            print("Failed dates:", failed)
    else:
        ap.error("provide --date, --dates-file, or --start/--end")
