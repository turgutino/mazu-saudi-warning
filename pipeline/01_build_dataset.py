# =============================================================================
# MAZU Saudi Arabia — Phase 0: Data Pipeline  (fast netCDF4 reader)
#
# Reads the 365 daily indicator NetCDF files, extracts the teacher-approved
# "core reliable" indicators, regrids SST onto the main 160x220 grid, and
# writes ONE consolidated dataset (time x lat x lon) for downstream modeling.
#
# Uses netCDF4 for reading (selective, ~0.05 s/file) instead of xarray
# (which parses all 91 variables on open, ~13 s/file).
#
# Input : E:\Data\New data\indicators\saudi_indicators_YYYYMMDD.nc  (365 files)
# Output: mazu-system\data\mazu_dataset.nc  +  dataset_report.txt
# =============================================================================

import os
import glob
import sys
import datetime as dt
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import warnings

warnings.filterwarnings("ignore")

IN_DIR  = r"E:\Data\New data\indicators"
OUT_DIR = r"C:\Users\Turqut\Desktop\Competation\mazu-system\data"
OUT_NC  = os.path.join(OUT_DIR, "mazu_dataset.nc")
OUT_RPT = os.path.join(OUT_DIR, "dataset_report.txt")

FEATURES_2D = [
    "daily_precip_total", "daily_convective_precip", "daily_large_scale_precip",
    "t2m_c", "tmax_c", "tmin_c", "heat_index_c", "vpd_kpa",
    "cape", "pwat", "ivt", "wind850_speed", "wind_shear_850_200",
    "daily_precip_anomaly", "t2m_anomaly_c", "tmax_anomaly_c",
]
FEATURE_SST = "sst_celsius"
LABELS = ["flash_flood_risk", "heatwave_day_flag", "heatwave_duration_days"]

N_LAT, N_LON = 160, 220


def parse_date(fname):
    s = os.path.basename(fname).replace("saudi_indicators_", "").replace(".nc", "")
    return dt.datetime.strptime(s, "%Y%m%d")


def to_nan(arr):
    """Convert a possibly-masked netCDF4 array to float32 with NaN fill."""
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype="float32")


def read_file(path):
    """Return dict {var: 2D (160,220) float32 array or None} using netCDF4."""
    out = {}
    nc = Dataset(path)
    try:
        for v in FEATURES_2D + LABELS:
            if v in nc.variables and nc.variables[v].shape == (N_LAT, N_LON):
                out[v] = to_nan(nc.variables[v][:])
            else:
                out[v] = None
        # SST: (time,lat,lon)=(4,160,221) -> time-mean, flip lat, slice lon 220
        if FEATURE_SST in nc.variables:
            sst = to_nan(nc.variables[FEATURE_SST][:])   # (4,160,221)
            if sst.ndim == 3:
                sst = np.nanmean(sst, axis=0)            # (160,221)
            sst = sst[::-1, :N_LON]                       # flip lat -> descending, 220 lon
            out[FEATURE_SST] = sst.astype("float32")
        else:
            out[FEATURE_SST] = None
    finally:
        nc.close()
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(IN_DIR, "saudi_indicators_*.nc")))
    print(f"Fayl sayi: {len(files)}", flush=True)
    if not files:
        print("[ERROR] Fayl yoxdur"); sys.exit(1)

    # reference grid from first file (via xarray, one-time)
    ref = xr.open_dataset(files[0])
    target_lat = ref["latitude"].values
    target_lon = ref["longitude"].values
    ref.close()

    dates = []
    allvars = FEATURES_2D + [FEATURE_SST] + LABELS
    store = {v: [] for v in allvars}
    valid_days = {v: 0 for v in allvars}
    empty = np.full((N_LAT, N_LON), np.nan, dtype="float32")

    t0 = dt.datetime.now()
    for i, f in enumerate(files):
        dates.append(parse_date(f))
        rec = read_file(f)
        for v in allvars:
            a = rec.get(v)
            if a is None:
                store[v].append(empty.copy())
            else:
                store[v].append(a)
                if np.isfinite(a).any():
                    valid_days[v] += 1
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{len(files)}  ({(dt.datetime.now()-t0).total_seconds():.1f}s)", flush=True)

    # ── consolidated dataset ────────────────────────────────────────────
    times = np.array([np.datetime64(d) for d in dates])
    data_vars = {v: (("time", "latitude", "longitude"),
                     np.stack(store[v], axis=0)) for v in allvars}
    out = xr.Dataset(data_vars,
                     coords={"time": times, "latitude": target_lat, "longitude": target_lon})
    out.attrs["title"] = "MAZU Saudi consolidated indicator dataset"
    out.attrs["features_2d"] = ",".join(FEATURES_2D)
    out.attrs["feature_sst"] = FEATURE_SST
    out.attrs["labels"] = ",".join(LABELS)

    enc = {v: {"zlib": True, "complevel": 4} for v in data_vars}
    out.to_netcdf(OUT_NC, encoding=enc)
    size_mb = os.path.getsize(OUT_NC) / 1024 / 1024

    # ── report ──────────────────────────────────────────────────────────
    L = ["=" * 62, "MAZU DATASET — Phase 0 report", "=" * 62,
         f"Files processed : {len(files)}",
         f"Date range      : {dates[0].date()} -> {dates[-1].date()}",
         f"Grid            : {N_LAT} x {N_LON} (0.1 deg)",
         f"Elapsed read    : {(dt.datetime.now()-t0).total_seconds():.1f}s",
         f"Output          : {OUT_NC} ({size_mb:.1f} MB)", "",
         f"{'variable':30s} {'valid_days':>10s} {'group':>8s}", "-" * 52]
    for v in FEATURES_2D:
        L.append(f"{v:30s} {valid_days[v]:>10d} {'feature':>8s}")
    L.append(f"{FEATURE_SST:30s} {valid_days[FEATURE_SST]:>10d} {'feature':>8s}")
    for v in LABELS:
        L.append(f"{v:30s} {valid_days[v]:>10d} {'label':>8s}")
    L += ["", "Sanity (mean/min/max over valid cells):"]
    for v in ["daily_precip_total", "tmax_c", "cape", "ivt", "sst_celsius", "flash_flood_risk"]:
        a = out[v].values; vv = a[np.isfinite(a)]
        if vv.size:
            L.append(f"  {v:28s} mean={vv.mean():10.3f} min={vv.min():10.3f} max={vv.max():10.3f}")
    rpt = "\n".join(L)
    with open(OUT_RPT, "w", encoding="utf-8") as fh:
        fh.write(rpt)
    print("\n" + rpt, flush=True)
    out.close()
    print(f"\n[DONE] {OUT_NC}", flush=True)


if __name__ == "__main__":
    main()
