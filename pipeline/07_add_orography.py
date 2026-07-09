# =============================================================================
# MAZU — Extension: add static orography (elevation) to the consolidated dataset
#
# Needed for the terrain-context / physical-plausibility check added to
# forecast_tool (idea sourced from reviewing a related team's discussion of
# GraphCast's known high-elevation error mode: forecasts should be flagged as
# lower-confidence in mountain terrain, per that discussion). Orography is
# static (time-invariant), so we pull it once from a single raw file rather
# than the full 365-day series.
# =============================================================================
import os
import xarray as xr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = r"E:\Data\New data\indicators"
CONSOLIDATED = os.path.join(HERE, "..", "data", "mazu_dataset.nc")

SAMPLE_RAW = os.path.join(RAW_DIR, "saudi_indicators_20250823.nc")

cons = xr.open_dataset(CONSOLIDATED)
raw = xr.open_dataset(SAMPLE_RAW)

assert np.array_equal(cons.latitude.values, raw.latitude.values), "latitude grid mismatch"
assert np.array_equal(cons.longitude.values, raw.longitude.values), "longitude grid mismatch"
assert "orography" in raw, "orography variable missing from raw source"

orog = raw["orography"].astype("float32")
cons = cons.assign(orography=(("latitude", "longitude"), orog.values))
cons["orography"].attrs = {"units": "m", "long_name": "surface elevation (static, from raw source)"}

out_path = CONSOLIDATED
cons.to_netcdf(out_path + ".tmp")
cons.close()
raw.close()
os.replace(out_path + ".tmp", out_path)
print(f"Added 'orography' variable to {out_path}")

check = xr.open_dataset(out_path)
assert "orography" in check
print("Verified: orography present, shape", check["orography"].shape,
      "min/max", float(check["orography"].min()), float(check["orography"].max()))
check.close()
