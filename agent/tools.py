# =============================================================================
# MAZU — Layer 4: Agent tools
#
# Three independently-testable tools the LLM agent can call:
#   forecast_tool(city, date, hazard)   -> tomorrow's risk probability
#   causal_kg_tool(hazard)              -> driving mechanisms + literature citations
#   conditions_tool(city, date)         -> today's actual raw indicator values
#
# Scope note (disclosed): this operates on the 2025 historical dataset only.
# "date" simulates "today" within that dataset; there is no live/real-time
# feed. This is an honest demo-scale agent, not an operational system.
# =============================================================================

import os
import importlib.util
import json
import joblib
import numpy as np
import xarray as xr
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "model")
SAVED_DIR = os.path.join(HERE, "saved_models")
DATASET = os.path.join(HERE, "..", "data", "mazu_dataset.nc")
KG_JSON = os.path.join(HERE, "..", "kg", "kg_data.json")

CITIES = {
    "Jeddah": (21.5, 39.2), "Mecca": (21.4, 39.8), "Riyadh": (24.7, 46.7),
    "Jizan": (16.9, 42.6), "Dammam": (26.4, 50.1), "Taif": (21.3, 40.4),
    "Medina": (24.5, 39.6), "Abha": (18.2, 42.5),
}

_bn_spec = importlib.util.spec_from_file_location("bn", os.path.join(MODEL_DIR, "06_baseline_with_neighbors.py"))
_bn = importlib.util.module_from_spec(_bn_spec)
_bn_src = open(os.path.join(MODEL_DIR, "06_baseline_with_neighbors.py"), encoding="utf-8").read()
_bn_src = _bn_src.replace('if __name__ == "__main__":\n    main()', "")
exec(compile(_bn_src, "bn", "exec"), _bn.__dict__)
neighbor_mean = _bn.neighbor_mean
NEIGHBOR_VARS = _bn.NEIGHBOR_VARS

FEATURE_VARS = [
    "daily_precip_total", "daily_convective_precip", "daily_large_scale_precip",
    "t2m_c", "tmax_c", "tmin_c", "heat_index_c", "vpd_kpa",
    "cape", "pwat", "ivt", "wind850_speed", "wind_shear_850_200",
    "daily_precip_anomaly", "t2m_anomaly_c", "tmax_anomaly_c", "sst_celsius",
]

_MODELS = {}
_META = None
_DS = None


def _load_resources():
    global _META, _DS
    if _META is None:
        with open(os.path.join(SAVED_DIR, "model_meta.json"), encoding="utf-8") as f:
            _META = json.load(f)
    if _DS is None:
        _DS = xr.open_dataset(DATASET)
    return _META, _DS


def _get_model(hazard):
    if hazard not in _MODELS:
        _MODELS[hazard] = joblib.load(os.path.join(SAVED_DIR, f"{hazard}_model.joblib"))
    return _MODELS[hazard]


def _nearest_stride2_index(lat_arr_s, lon_arr_s, lat, lon):
    yi = int(np.argmin(np.abs(lat_arr_s - lat)))
    xi = int(np.argmin(np.abs(lon_arr_s - lon)))
    return yi, xi


# =============================================================================
# TOOL 1: forecast
# =============================================================================

def forecast_tool(city: str, target_date: str, hazard: str) -> dict:
    """Predict hazard risk for `target_date`, using the PREVIOUS day's indicators
    (genuine t-1 -> t forecasting, matching Layer 2's verified methodology).

    Design note: earlier testing found that having the agent pass "today's date"
    and mentally computing "predicts tomorrow" was an unnecessary source of
    off-by-one errors when a user asks "what's the risk ON date X" -- the LLM
    would sometimes pass X itself as "today", yielding a forecast for X+1
    instead of X. This tool now takes `target_date` (the date being forecast)
    directly and internally looks up target_date-1 for input features, removing
    that entire class of agent reasoning error.

    Args:
        city: one of CITIES keys (e.g. "Jeddah")
        target_date: "YYYY-MM-DD" -- the date whose risk is being forecast
        hazard: "heatwave" or "flash_flood"
    Returns dict with probability, the date used for input features, and errors.
    """
    if city not in CITIES:
        return {"error": f"Unknown city '{city}'. Known cities: {list(CITIES.keys())}"}
    if hazard not in ("heatwave", "flash_flood"):
        return {"error": f"Unknown hazard '{hazard}'. Must be 'heatwave' or 'flash_flood'."}

    meta, ds = _load_resources()
    times = np.array([str(t)[:10] for t in ds.time.values])
    if target_date not in times:
        return {"error": f"Date '{target_date}' not in dataset range (2025-01-01 to 2025-12-31)."}
    ti_target = int(np.where(times == target_date)[0][0])
    if ti_target - 1 < 0:
        return {"error": f"No prior-day data available before '{target_date}' (start of dataset)."}
    ti = ti_target - 1
    date_predicted = times[ti_target]

    lat_full, lon_full = ds.latitude.values, ds.longitude.values
    stride = meta["stride"]
    yi_s = np.arange(0, len(lat_full), stride)
    xi_s = np.arange(0, len(lon_full), stride)
    lat_s, lon_s = lat_full[yi_s], lon_full[xi_s]

    city_lat, city_lon = CITIES[city]
    cyi, cxi = _nearest_stride2_index(lat_s, lon_s, city_lat, city_lon)

    doy = ds.time.values[ti].astype("datetime64[D]").item().timetuple().tm_yday

    raw = {}
    for v in FEATURE_VARS:
        arr = ds[v].values[ti][yi_s][:, xi_s]   # stride-2 2D field for this day
        raw[v] = arr
    feat_row = [raw[v][cyi, cxi] for v in FEATURE_VARS]

    if hazard == "heatwave":
        for v in NEIGHBOR_VARS:
            nm = neighbor_mean(raw[v])
            feat_row.append(nm[cyi, cxi])
    feat_row += [city_lat, city_lon, doy]

    X = np.array(feat_row, dtype="float64").reshape(1, -1)
    model = _get_model(hazard)
    proba = float(model.predict_proba(X)[0, 1])

    return {
        "city": city, "target_date": date_predicted, "features_from_date": times[ti],
        "hazard": hazard, "probability": round(proba, 4),
        "grid_cell": {"lat": round(float(lat_s[cyi]), 2), "lon": round(float(lon_s[cxi]), 2)},
        "model_verified_roc_auc": meta[hazard]["roc_auc"],
    }


# =============================================================================
# TOOL 2: causal knowledge graph query
# =============================================================================

def causal_kg_tool(hazard: str) -> dict:
    """Return the driving mechanisms for a hazard and their literature grounding.

    Args:
        hazard: one of the Hazard node ids in the KG, e.g. "flash_flood", "heatwave"
    """
    with open(KG_JSON, encoding="utf-8") as f:
        kg = json.load(f)
    node_by_id = {n["id"]: n for n in kg["nodes"]}
    if hazard not in node_by_id:
        return {"error": f"Unknown hazard '{hazard}'. Known: "
                         f"{[n['id'] for n in kg['nodes'] if n.get('ntype')=='Hazard']}"}

    # edge direction (verified against kg/01_build_structural_kg.py): hazard --driven_by--> mechanism
    mechanisms = [l["target"] for l in kg["links"]
                 if l.get("etype") == "driven_by" and l["source"] == hazard]
    indicators = [l["source"] for l in kg["links"]
                 if l.get("etype") == "contributes_to" and l["target"] == hazard]

    mech_info = []
    for m in mechanisms:
        citations = [l["target"] for l in kg["links"]
                    if l.get("etype") == "grounded_by" and l["source"] == m]
        cit_details = []
        for c in citations:
            cnode = node_by_id.get(c, {})
            cit_details.append({
                "citation": cnode.get("label"), "url": cnode.get("url"),
                "evidence": cnode.get("evidence", [])[:2],   # cap for brevity
            })
        mech_info.append({
            "mechanism": m, "description": node_by_id[m].get("desc", ""),
            "literature_grounded": len(cit_details) > 0,
            "citations": cit_details,
        })

    return {
        "hazard": hazard, "contributing_indicators": indicators,
        "mechanisms": mech_info,
    }


# =============================================================================
# TOOL 3: current conditions readout
# =============================================================================

def conditions_tool(city: str, date: str) -> dict:
    """Return today's actual raw indicator values for a city (ground truth context)."""
    if city not in CITIES:
        return {"error": f"Unknown city '{city}'. Known cities: {list(CITIES.keys())}"}
    meta, ds = _load_resources()
    times = np.array([str(t)[:10] for t in ds.time.values])
    if date not in times:
        return {"error": f"Date '{date}' not in dataset range (2025-01-01 to 2025-12-31)."}
    ti = int(np.where(times == date)[0][0])

    lat_full, lon_full = ds.latitude.values, ds.longitude.values
    city_lat, city_lon = CITIES[city]
    yi = int(np.argmin(np.abs(lat_full - city_lat)))
    xi = int(np.argmin(np.abs(lon_full - city_lon)))

    values = {}
    for v in FEATURE_VARS:
        val = float(ds[v].values[ti, yi, xi])
        if np.isfinite(val):
            values[v] = round(val, 2)
    return {"city": city, "date": date,
            "grid_cell": {"lat": round(float(lat_full[yi]), 2), "lon": round(float(lon_full[xi]), 2)},
            "indicators": values}


if __name__ == "__main__":
    print("tools.py loaded OK — run test_tools.py for verification.")
