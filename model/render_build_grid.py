import sys, os, json
sys.path.insert(0, r"C:\Users\Turqut\Desktop\Competation\mazu-system\agent")
import numpy as np
import tools

def predicted_grid(date, hazard):
    """Mirrors forecast_tool's feature construction exactly, but vectorized
    across the whole stride-2 grid instead of one city, using date-1 inputs."""
    meta, ds = tools._load_resources()
    times = np.array([str(t)[:10] for t in ds.time.values])
    ti_target = int(np.where(times == date)[0][0])
    ti = ti_target - 1
    if ti < 0:
        raise ValueError("no prior day")

    lat_full, lon_full = ds.latitude.values, ds.longitude.values
    stride = meta["stride"]
    yi_s = np.arange(0, len(lat_full), stride)
    xi_s = np.arange(0, len(lon_full), stride)
    lat_s, lon_s = lat_full[yi_s], lon_full[xi_s]
    ny, nx = len(lat_s), len(lon_s)

    doy = ds.time.values[ti].astype("datetime64[D]").item().timetuple().tm_yday

    raw = {}
    for v in tools.FEATURE_VARS:
        raw[v] = ds[v].values[ti][yi_s][:, xi_s]

    cols = [raw[v].reshape(-1) for v in tools.FEATURE_VARS]

    if hazard == "heatwave":
        for v in tools.NEIGHBOR_VARS:
            nm = tools.neighbor_mean(raw[v])
            cols.append(nm.reshape(-1))
    elif hazard == "dust_storm":
        for v in tools.DUST_EXTRA_VARS:
            arr = ds[v].values[ti][yi_s][:, xi_s]
            cols.append(arr.reshape(-1))

    lat_grid, lon_grid = np.meshgrid(lat_s, lon_s, indexing="ij")
    cols.append(lat_grid.reshape(-1))
    cols.append(lon_grid.reshape(-1))
    cols.append(np.full(ny * nx, doy, dtype="float64"))

    X = np.stack(cols, axis=1).astype("float64")

    model = tools._get_model(hazard)
    proba = model.predict_proba(X)[:, 1]
    proba = proba.reshape(ny, nx)
    return lat_s, lon_s, proba


def _build_feature_matrix(date, hazard):
    """Shared feature construction, factored out of predicted_grid so the
    ensemble-uncertainty grid can reuse it without duplicating the logic."""
    meta, ds = tools._load_resources()
    times = np.array([str(t)[:10] for t in ds.time.values])
    ti_target = int(np.where(times == date)[0][0])
    ti = ti_target - 1
    if ti < 0:
        raise ValueError("no prior day")

    lat_full, lon_full = ds.latitude.values, ds.longitude.values
    stride = meta["stride"]
    yi_s = np.arange(0, len(lat_full), stride)
    xi_s = np.arange(0, len(lon_full), stride)
    lat_s, lon_s = lat_full[yi_s], lon_full[xi_s]
    ny, nx = len(lat_s), len(lon_s)

    doy = ds.time.values[ti].astype("datetime64[D]").item().timetuple().tm_yday

    raw = {}
    for v in tools.FEATURE_VARS:
        raw[v] = ds[v].values[ti][yi_s][:, xi_s]

    cols = [raw[v].reshape(-1) for v in tools.FEATURE_VARS]

    if hazard == "heatwave":
        for v in tools.NEIGHBOR_VARS:
            nm = tools.neighbor_mean(raw[v])
            cols.append(nm.reshape(-1))
    elif hazard == "dust_storm":
        for v in tools.DUST_EXTRA_VARS:
            arr = ds[v].values[ti][yi_s][:, xi_s]
            cols.append(arr.reshape(-1))

    lat_grid, lon_grid = np.meshgrid(lat_s, lon_s, indexing="ij")
    cols.append(lat_grid.reshape(-1))
    cols.append(lon_grid.reshape(-1))
    cols.append(np.full(ny * nx, doy, dtype="float64"))

    X = np.stack(cols, axis=1).astype("float64")
    return X, lat_s, lon_s, ny, nx


def predicted_grid_uncertainty(date, hazard):
    """Grid-wide ensemble-spread uncertainty, using the same 5-member
    production ensemble tools._ensemble_uncertainty() uses for single-point
    queries (models/ensemble/{hazard}_seed{42..46}.joblib, already trained
    -- see tools.py's _get_ensemble_models). Returns (lat, lon, std_grid)
    where std_grid is the per-cell standard deviation of the 5 members'
    predict_proba, i.e. exactly the "model confidence" signal the reviewer
    asked for: low std = members agree = confident; high std = members
    disagree = uncertain (e.g. sparse-data mountain/border cells)."""
    X, lat_s, lon_s, ny, nx = _build_feature_matrix(date, hazard)
    members = tools._get_ensemble_models(hazard)
    probs = np.stack([m.predict_proba(X)[:, 1] for m in members], axis=0)  # (n_members, ny*nx)
    std = probs.std(axis=0).reshape(ny, nx)
    return lat_s, lon_s, std


def real_grid(date, hazard):
    de = tools._get_detection_engine()
    risk = de.risk_field(date, hazard)
    return de.lat, de.lon, risk


if __name__ == "__main__":
    # sanity check: predicted_grid at Dammam's nearest cell on 2025-05-17 dust_storm
    # must equal forecast_tool('Dammam','2025-05-17','dust_storm') exactly.
    lat_s, lon_s, grid = predicted_grid("2025-05-17", "dust_storm")
    city_lat, city_lon = tools.CITIES["Dammam"]
    yi = int(np.argmin(np.abs(lat_s - city_lat)))
    xi = int(np.argmin(np.abs(lon_s - city_lon)))
    grid_val = round(float(grid[yi, xi]), 4)
    tool_val = round(tools.forecast_tool("Dammam", "2025-05-17", "dust_storm")["probability"], 4)
    print("grid value:", grid_val)
    print("forecast_tool value:", tool_val)
    print("MATCH:", grid_val == tool_val)

    # cross-check real_grid against DetectionEngine.detect()'s Dammam cluster (peak_risk 0.6)
    rlat, rlon, rgrid = real_grid("2025-05-17", "dust_storm")
    city_lat, city_lon = tools.CITIES["Dammam"]
    ryi = int(np.argmin(np.abs(rlat - city_lat)))
    rxi = int(np.argmin(np.abs(rlon - city_lon)))
    print("real_grid Dammam risk:", round(float(rgrid[ryi, rxi]), 3), "(detect() reported peak_risk 0.6 nearby)")
