# =============================================================================
# MAZU — Extension: add dust-storm-relevant indicators to the consolidated
# dataset (wind10_speed, dewpoint_depression_c). These are time-varying (one
# value per day, unlike orography), so all 365 raw files must be read --
# using the same fast netCDF4 selective-read pattern as 01_build_dataset.py
# (not xarray's full-file parse) to keep this fast.
# =============================================================================
import os
import glob
import datetime as dt
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = r"E:\Data\New data\indicators"
CONSOLIDATED = os.path.join(HERE, "..", "data", "mazu_dataset.nc")

NEW_VARS = ["wind10_speed", "dewpoint_depression_c"]
N_LAT, N_LON = 160, 220


def parse_date(fname):
    s = os.path.basename(fname).replace("saudi_indicators_", "").replace(".nc", "")
    return dt.datetime.strptime(s, "%Y%m%d")


def to_nan(arr):
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype="float32")


cons = xr.open_dataset(CONSOLIDATED)
cons_times = np.array([str(t)[:10] for t in cons.time.values])
cons.close()

files = sorted(glob.glob(os.path.join(RAW_DIR, "saudi_indicators_*.nc")))
assert len(files) == 365, f"expected 365 raw files, found {len(files)}"

store = {v: [] for v in NEW_VARS}
empty = np.full((N_LAT, N_LON), np.nan, dtype="float32")
t0 = dt.datetime.now()
file_dates = []
for i, f in enumerate(files):
    file_dates.append(parse_date(f).strftime("%Y-%m-%d"))
    nc = Dataset(f)
    try:
        for v in NEW_VARS:
            if v in nc.variables and nc.variables[v].shape == (N_LAT, N_LON):
                store[v].append(to_nan(nc.variables[v][:]))
            else:
                store[v].append(empty.copy())
    finally:
        nc.close()
    if (i + 1) % 90 == 0:
        print(f"  {i+1}/365  ({(dt.datetime.now()-t0).total_seconds():.1f}s)", flush=True)

# raw files are already in the same date order as the consolidated dataset's
# time axis (both built by iterating sorted glob of the same filenames) --
# verify this explicitly rather than assuming it.
assert file_dates == list(cons_times), "raw file date order does not match consolidated dataset time axis"

cons2 = xr.open_dataset(CONSOLIDATED)
for v in NEW_VARS:
    cons2 = cons2.assign(**{v: (("time", "latitude", "longitude"), np.stack(store[v], axis=0))})
    cons2[v].attrs = {"units": "m s-1" if v == "wind10_speed" else "degC"}

tmp_path = CONSOLIDATED + ".tmp"
cons2.to_netcdf(tmp_path)
cons2.close()
os.replace(tmp_path, CONSOLIDATED)
print(f"Added {NEW_VARS} to {CONSOLIDATED}")

check = xr.open_dataset(CONSOLIDATED)
for v in NEW_VARS:
    arr = check[v].values
    valid = np.isfinite(arr)
    print(f"  {v}: valid_days~{valid.any(axis=(1,2)).sum()}/365, "
         f"mean={np.nanmean(arr):.2f}, min={np.nanmin(arr):.2f}, max={np.nanmax(arr):.2f}")
check.close()
