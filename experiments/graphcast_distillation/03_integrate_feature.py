# =============================================================================
# Regrid the per-day GraphCast (1deg) outputs onto our 0.1deg CMA grid and add
# them as new feature columns. Runs LOCALLY (no GPU needed -- this is just
# xarray/numpy regridding of small, already-cropped files).
#
# Never overwrites the original mazu_dataset.nc -- writes a new file.
#
# Regridding method: nearest-neighbor. GraphCast Small is coarser (1deg,
# ~111km) than our CMA grid (0.1deg, ~11km), so any interpolation beyond
# nearest-neighbor would fabricate precision the source data doesn't have.
# Every CMA cell within a given 1deg GraphCast cell gets that same GraphCast
# value -- a deliberately honest, blocky field, not a smoothed one.
# =============================================================================
import glob
import os

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw_graphcast_daily")
ORIGINAL_DATASET = os.path.join(HERE, "..", "..", "..", "data", "mazu_dataset.nc")
OUTPUT_DATASET = os.path.join(HERE, "mazu_dataset_with_graphcast.nc")

NEW_FEATURE_VARS = ["t2m", "u10m", "v10m", "msl", "tp06"]  # tp06 confirmed as the
# real GraphCastSmall output var name via a live run, 2026-07-19 (see 02_run_graphcast_inference.py)
RENAMED = {  # avoid colliding with existing CMA-derived variable names
    "t2m": "gc_t2m",
    "u10m": "gc_u10m",
    "v10m": "gc_v10m",
    "msl": "gc_msl",
    "tp06": "gc_precip",
}


def nearest_neighbor_regrid(gc_da: xr.DataArray, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    """Map each target (CMA) grid cell to its nearest GraphCast (1deg) cell.
    Returns a (len(target_lat), len(target_lon)) array."""
    src_lat = gc_da["lat"].values
    src_lon = gc_da["lon"].values
    lat_idx = np.array([np.argmin(np.abs(src_lat - la)) for la in target_lat])
    lon_idx = np.array([np.argmin(np.abs(src_lon - lo)) for lo in target_lon])
    return gc_da.values[np.ix_(lat_idx, lon_idx)]


def main():
    if not os.path.isdir(RAW_DIR) or not glob.glob(os.path.join(RAW_DIR, "graphcast_*.nc")):
        raise SystemExit(
            f"No GraphCast daily files found in {RAW_DIR}. "
            "Run 02_run_graphcast_inference.py on the server first, then copy "
            "raw_graphcast_daily/ back to this machine before running this script."
        )

    ds = xr.open_dataset(ORIGINAL_DATASET)
    times = np.array([str(t)[:10] for t in ds.time.values])
    lat, lon = ds.latitude.values, ds.longitude.values

    new_arrays = {new_name: np.full((len(times), len(lat), len(lon)), np.nan, dtype="float32")
                  for new_name in RENAMED.values()}

    n_found, n_missing = 0, 0
    for ti, date_str in enumerate(times):
        fpath = os.path.join(RAW_DIR, f"graphcast_{date_str}.nc")
        if not os.path.exists(fpath):
            n_missing += 1
            continue
        gc = xr.open_dataset(fpath)
        for var, new_name in RENAMED.items():
            new_arrays[new_name][ti] = nearest_neighbor_regrid(gc[var].isel(time=0) if "time" in gc[var].dims else gc[var], lat, lon)
        gc.close()
        n_found += 1

    print(f"Integrated {n_found} days, {n_missing} missing (left as NaN -- HistGradientBoostingClassifier handles NaN natively).")

    for new_name, arr in new_arrays.items():
        ds[new_name] = (("time", "latitude", "longitude"), arr)

    ds.to_netcdf(OUTPUT_DATASET)
    ds.close()
    print(f"[SAVED] {OUTPUT_DATASET} (original mazu_dataset.nc untouched)")


if __name__ == "__main__":
    main()
