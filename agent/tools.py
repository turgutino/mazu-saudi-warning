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

# Layer 1's rule-based detection engine (RULES: data-grounded percentile
# thresholds per hazard, verified in 01_detection_engine.py's own test suite)
# is reused here as an INDEPENDENT cross-check signal for the ML forecast --
# see _reflexive_check() below.
_de_spec = importlib.util.spec_from_file_location("de", os.path.join(MODEL_DIR, "01_detection_engine.py"))
_de = importlib.util.module_from_spec(_de_spec)
_de_src = open(os.path.join(MODEL_DIR, "01_detection_engine.py"), encoding="utf-8").read()
_de_src = _de_src.split('if __name__ == "__main__":')[0]
exec(compile(_de_src, "de", "exec"), _de.__dict__)
DetectionEngine = _de.DetectionEngine
DETECTION_RULES = _de.RULES

FEATURE_VARS = [
    "daily_precip_total", "daily_convective_precip", "daily_large_scale_precip",
    "t2m_c", "tmax_c", "tmin_c", "heat_index_c", "vpd_kpa",
    "cape", "pwat", "ivt", "wind850_speed", "wind_shear_850_200",
    "daily_precip_anomaly", "t2m_anomaly_c", "tmax_anomaly_c", "sst_celsius",
]
# dust_storm's model was trained with these 2 extra features appended after
# the base FEATURE_VARS (see model/07_dust_storm_forecast.py) -- order matters.
DUST_EXTRA_VARS = ["wind10_speed", "dewpoint_depression_c"]

_MODELS = {}
_ENSEMBLES = {}
_META = None
_DS = None
_POP = None
_DE = None
ENSEMBLE_DIR = os.path.join(SAVED_DIR, "ensemble")


def _load_resources():
    global _META, _DS
    if _META is None:
        with open(os.path.join(SAVED_DIR, "model_meta.json"), encoding="utf-8") as f:
            _META = json.load(f)
    if _DS is None:
        _DS = xr.open_dataset(DATASET)
    return _META, _DS


def _load_population():
    global _POP
    if _POP is None:
        with open(os.path.join(HERE, "city_population.json"), encoding="utf-8") as f:
            _POP = json.load(f)
    return _POP


def _get_model(hazard):
    if hazard not in _MODELS:
        _MODELS[hazard] = joblib.load(os.path.join(SAVED_DIR, f"{hazard}_model.joblib"))
    return _MODELS[hazard]


# 5-member bagging ensembles (model/13_train_ensemble_members.py), same
# hyperparameters/training window as the single production model above,
# differing only in random_state (42-46; seed 42 is trained identically to
# the production model, i.e. a 6th, redundant copy of it -- included
# deliberately for symmetry/simplicity, not because it adds information).
# Used ONLY to compute an ensemble-spread uncertainty estimate around the
# production model's own point probability -- these members' predictions are
# never substituted for the production model's probability itself.
_ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]


def _get_ensemble_models(hazard):
    if hazard not in _ENSEMBLES:
        members = []
        for seed in _ENSEMBLE_SEEDS:
            path = os.path.join(ENSEMBLE_DIR, f"{hazard}_seed{seed}.joblib")
            members.append(joblib.load(path))
        _ENSEMBLES[hazard] = members
    return _ENSEMBLES[hazard]


def _ensemble_uncertainty(hazard, X):
    """Returns (mean_proba, uncertainty_dict) from the 5-member ensemble's
    spread on this single feature row. mean_proba is reported separately from
    (not used to replace) the production model's own point probability --
    see forecast_tool, which keeps the two clearly distinct."""
    members = _get_ensemble_models(hazard)
    probs = np.array([float(m.predict_proba(X)[0, 1]) for m in members])
    return {
        "mean": round(float(probs.mean()), 4),
        "std": round(float(probs.std()), 4),
        "range": [round(float(probs.min()), 4), round(float(probs.max()), 4)],
        "n_members": len(members),
    }


def _get_detection_engine():
    global _DE
    if _DE is None:
        _DE = DetectionEngine(dataset=DATASET)
    return _DE


# Threshold at which each independent signal (ML probability, rule-based
# detection risk score) counts as "elevated". Both are 0..1 scores but are NOT
# directly comparable in a calibrated sense (one is a trained classifier's
# probability, the other a weighted-condition score) -- 0.3 is a deliberately
# moderate, round cutoff chosen to sit clearly above typical rare-event base
# rates (~0.5-5%) while staying below the detection engine's own event-cluster
# threshold (0.5-0.55, see RULES in 01_detection_engine.py), so this flags
# meaningful signal without requiring a full clustered event. It is a fixed,
# documented choice, not tuned to produce any particular result.
_REFLEXIVE_THRESHOLD = 0.3


def _reflexive_check(city_lat, city_lon, features_date, hazard, model_proba):
    """Cross-check the ML model's probability against Layer 1's INDEPENDENT
    rule-based detection engine, evaluated on the SAME day's raw indicators
    the model used as input (features_date). This is a Reflexion-style
    self-consistency check (cf. Shinn et al. 2023; used in the same spirit by
    MAESTRO, npj Artificial Intelligence 2026, to validate agent outputs
    against independent domain constraints) -- two independently-built
    signals (a trained classifier vs. hand-set, data-grounded physical
    thresholds) should broadly agree; when they don't, that disagreement is
    itself useful information, not something to hide.
    """
    eng = _get_detection_engine()
    if features_date not in eng.times:
        return None
    risk_field = eng.risk_field(features_date, hazard)
    yi = int(np.argmin(np.abs(eng.lat - city_lat)))
    xi = int(np.argmin(np.abs(eng.lon - city_lon)))
    detection_risk = float(risk_field[yi, xi])

    rule = DETECTION_RULES[hazard]
    ti = int(np.where(eng.times == features_date)[0][0])
    fired = []
    for c in rule["conditions"]:
        if c["ind"] in eng.ds:
            v = float(eng.ds[c["ind"]].values[ti, yi, xi])
            op = _de._OPS[c["op"]]
            if np.isfinite(v) and op(v, c["thr"]):
                fired.append(f"{c['ind']}={v:.1f}")

    model_elevated = model_proba >= _REFLEXIVE_THRESHOLD
    detection_elevated = detection_risk >= _REFLEXIVE_THRESHOLD

    if model_elevated and detection_elevated:
        consistency = "consistent_elevated"
        note = ("Model risk and independent rule-based physical indicators AGREE this is "
                "an elevated-risk situation.")
    elif not model_elevated and not detection_elevated:
        consistency = "consistent_low"
        note = ("Model risk and independent rule-based physical indicators AGREE this is "
                "a low-risk situation.")
    elif model_elevated and not detection_elevated:
        consistency = "model_higher_than_detection"
        note = ("Model flags elevated risk, but the classic physical drivers for this "
                "hazard (per Layer 1's data-grounded thresholds) were NOT present in that "
                "day's indicators at this location -- treat with added caution, the model "
                "may be picking up a signal the rule-based check doesn't capture, or may "
                "be over-confident.")
    else:
        consistency = "detection_higher_than_model"
        note = ("The classic physical drivers for this hazard WERE present that day, but "
                "the model assigned low probability -- also worth caution, this is the "
                "riskier mismatch direction for a genuine miss.")

    return {
        "model_probability": round(model_proba, 4),
        "detection_engine_risk_score": round(detection_risk, 4),
        "detection_engine_conditions_fired": fired,
        "threshold_used": _REFLEXIVE_THRESHOLD,
        "consistency": consistency,
        "note": note,
    }


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
        hazard: "heatwave", "flash_flood", or "dust_storm"
    Returns dict with probability, the date used for input features, and errors.
    """
    if city not in CITIES:
        return {"error": f"Unknown city '{city}'. Known cities: {list(CITIES.keys())}"}
    if hazard not in ("heatwave", "flash_flood", "dust_storm"):
        return {"error": f"Unknown hazard '{hazard}'. Must be 'heatwave', 'flash_flood', or 'dust_storm'."}

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
    elif hazard == "dust_storm":
        # matches model/07_dust_storm_forecast.py's FEATURE_VARS order exactly
        # (base 17 vars, then these 2 dust-specific vars, then lat/lon/doy)
        for v in DUST_EXTRA_VARS:
            arr = ds[v].values[ti][yi_s][:, xi_s]
            feat_row.append(arr[cyi, cxi])
    feat_row += [city_lat, city_lon, doy]

    X = np.array(feat_row, dtype="float64").reshape(1, -1)
    model = _get_model(hazard)
    proba = float(model.predict_proba(X)[0, 1])
    uncertainty = _ensemble_uncertainty(hazard, X)

    # Use FULL-resolution orography (not the model's stride-2 feature grid) for
    # elevation context: terrain is a static geographic fact and should reflect
    # the city's true elevation, not whichever coarse grid cell the model's
    # subsampled features happen to land nearest to. Testing found these can
    # diverge sharply in steep terrain (Abha: stride-2 cell = 1334m vs the
    # city's true ~2082m one full-res cell away) -- using the coarse value here
    # would understate exactly the mountain-terrain caution this field exists
    # to raise.
    elevation_m = None
    terrain_note = None
    if "orography" in ds:
        fyi = int(np.argmin(np.abs(lat_full - city_lat)))
        fxi = int(np.argmin(np.abs(lon_full - city_lon)))
        elevation_m = float(ds["orography"].values[fyi, fxi])
        if elevation_m >= 1500:
            terrain_note = (
                f"This city is at {round(elevation_m)} m elevation (mountain terrain, "
                f"e.g. Asir range). Meteorological indicators and model training data are "
                f"sparser here than in the low-elevation interior/coast, and the model's "
                f"feature grid is coarser than local terrain relief, so treat this "
                f"probability with added caution."
            )

    reflexive_check = _reflexive_check(city_lat, city_lon, times[ti], hazard, proba)

    pop = _load_population()
    city_pop = pop["cities"].get(city)
    impact_context = None
    if city_pop is not None:
        impact_context = {
            "city_population_2022_census": city_pop,
            "source": "Saudi Census 2022, GASTAT",
            "note": ("Reference population only, NOT a model output and NOT an estimate "
                     "of how many people would actually be exposed to this hazard -- this "
                     "system does not perform exposure/vulnerability modeling. Provided so "
                     "risk can be discussed in human terms, per WMO impact-based warning "
                     "guidance, not to imply a precise affected-population count."),
        }

    return {
        "city": city, "target_date": date_predicted, "features_from_date": times[ti],
        "hazard": hazard, "probability": round(proba, 4),
        "grid_cell": {"lat": round(float(lat_s[cyi]), 2), "lon": round(float(lon_s[cxi]), 2)},
        "elevation_m": round(elevation_m, 1) if elevation_m is not None else None,
        "terrain_note": terrain_note,
        "impact_context": impact_context,
        "reflexive_check": reflexive_check,
        "model_verified_roc_auc": meta[hazard]["roc_auc"],
        # WMO/national-met-service-standard verification metrics, computed at
        # this hazard's own operational alert threshold (same value CAP
        # severity uses -- see model/08_meteorological_metrics.py). These are
        # THRESHOLD-DEPENDENT, unlike ROC-AUC/PR-AUC above -- read together,
        # not as a replacement: a hazard can have strong ROC-AUC (good
        # discrimination across all thresholds) yet weak POD/CSI at this
        # fixed operational cutoff for genuinely rare events (see
        # flash_flood in agent/METEOROLOGICAL_METRICS_REPORT.md).
        "meteorological_metrics": meta[hazard]["meteorological_metrics"],
        # Ensemble-spread uncertainty (model/13_train_ensemble_members.py): 5
        # independently-trained classifiers (same hyperparameters/training
        # window as the production model, different random seeds) scored on
        # this exact feature row. "std"/"range" describe how much these 5
        # otherwise-identical models disagree with each other -- a MODEL-
        # DISAGREEMENT signal, not a statement that the true probability lies
        # in this range. Deliberately does NOT touch "probability" above
        # (the production model's own point estimate is reported unchanged,
        # exactly as before this field existed) -- this adds context around
        # the number rather than attempting to "fix" it, unlike the isotonic-
        # calibration/ensemble-averaging experiments in
        # agent/CALIBRATION_REPORT.md and agent/ENSEMBLE_REPORT.md, both of
        # which changed the point estimate itself and were found to hurt
        # operational alert quality.
        "uncertainty": uncertainty,
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


# =============================================================================
# TOOL 4: similar historical events
# =============================================================================

# Which raw indicators are compared per hazard. Chosen because they are the
# SAME variables already used elsewhere for this hazard (RULES in
# 01_detection_engine.py / FEATURE_VARS), not a new, untested feature set.
SIMILARITY_FEATURES = {
    "flash_flood": ["cape", "ivt", "pwat", "daily_precip_total", "wind850_speed"],
    "heatwave": ["tmax_c", "heat_index_c", "vpd_kpa", "t2m_c", "tmax_anomaly_c"],
    "dust_storm": ["wind10_speed", "wind850_speed", "dewpoint_depression_c", "vpd_kpa"],
}

_EVENTS = None
_FEATURE_STATS = {}


def _load_events():
    global _EVENTS
    if _EVENTS is None:
        with open(KG_JSON, encoding="utf-8") as f:
            kg = json.load(f)
        _EVENTS = [n for n in kg["nodes"] if n.get("ntype") == "Event"]
    return _EVENTS


def _feature_stats(var):
    """Dataset-wide mean/std for one variable, cached. Used to z-score before
    comparing, so variables on very different physical scales (e.g. cape in
    the thousands vs vpd_kpa in single digits) don't silently dominate the
    similarity score just because of their units."""
    if var not in _FEATURE_STATS:
        _, ds = _load_resources()
        arr = ds[var].values
        _FEATURE_STATS[var] = (float(np.nanmean(arr)), float(np.nanstd(arr)))
    return _FEATURE_STATS[var]


def _parse_event_location(loc_str):
    # "Jizan (17.5N,42.9E)" -> (17.5, 42.9)
    coords = loc_str.split("(")[1].rstrip(")").replace("N", "").replace("E", "")
    la, lo = coords.split(",")
    return float(la), float(lo)


def _get_vector(date, lat, lon, features):
    """Read raw indicator values at the nearest full-resolution grid cell for
    a date/location, for the given feature list. Returns dict{var: value or
    None if missing/out of range}."""
    _, ds = _load_resources()
    times = np.array([str(t)[:10] for t in ds.time.values])
    if date not in times:
        return None
    ti = int(np.where(times == date)[0][0])
    lat_full, lon_full = ds.latitude.values, ds.longitude.values
    yi = int(np.argmin(np.abs(lat_full - lat)))
    xi = int(np.argmin(np.abs(lon_full - lon)))
    vec = {}
    for v in features:
        val = float(ds[v].values[ti, yi, xi])
        vec[v] = val if np.isfinite(val) else None
    return vec


def similar_events_tool(city: str, date: str, hazard: str) -> dict:
    """Compare a city/date's actual indicator values against the KG's 5 known
    real 2025 extreme events (3 flash_flood, 2 heatwave, 1 dust_storm), ranked
    by similarity, so the agent can say "this looks like the 23 Aug event"
    instead of only stating a bare number.

    Similarity method (documented, not tuned to any specific outcome):
    z-score each shared feature using the dataset's own mean/std, then convert
    normalized Euclidean distance to a 0-100% score via 100/(1+distance) --
    distance 0 -> 100%, larger distance -> lower, monotonically decreasing.
    An event needs at least half its features available on both sides to be
    ranked; otherwise it is excluded with a reason (not silently dropped).
    """
    if city not in CITIES:
        return {"error": f"Unknown city '{city}'. Known cities: {list(CITIES.keys())}"}
    if hazard not in ("heatwave", "flash_flood", "dust_storm"):
        return {"error": f"Unknown hazard '{hazard}'. Must be 'heatwave', 'flash_flood', or 'dust_storm'."}
    features = SIMILARITY_FEATURES[hazard]

    city_lat, city_lon = CITIES[city]
    query_vec = _get_vector(date, city_lat, city_lon, features)
    if query_vec is None:
        return {"error": f"Date '{date}' not in dataset range (2025-01-01 to 2025-12-31)."}

    events = [e for e in _load_events() if e["hazard"] == hazard]
    results = []
    excluded = []
    for e in events:
        e_lat, e_lon = _parse_event_location(e["location"])
        event_vec = _get_vector(e["date"], e_lat, e_lon, features)
        if event_vec is None:
            excluded.append({"event": e["label"], "reason": "event date not in dataset"})
            continue

        sq_dists = []
        for v in features:
            qv, ev = query_vec.get(v), event_vec.get(v)
            if qv is None or ev is None:
                continue
            mean, std = _feature_stats(v)
            if std <= 0:
                continue
            sq_dists.append(((qv - mean) / std - (ev - mean) / std) ** 2)

        if len(sq_dists) < (len(features) + 1) // 2:
            excluded.append({"event": e["label"],
                             "reason": f"only {len(sq_dists)}/{len(features)} features available, "
                                       f"below the half-coverage minimum"})
            continue

        distance = float(np.sqrt(sum(sq_dists)))
        similarity_pct = round(100.0 / (1.0 + distance), 1)
        # Rough great-circle-ish distance (deg->km at ~111km/deg, adequate at
        # this latitude/precision for a disclosure caveat, not navigation).
        # Event coordinates in the KG are each event's own grid-cell MAXIMUM
        # (e.g. the exact storm centroid), which is commonly tens of km from
        # a city's center -- discovered while testing this tool (Jizan city
        # center vs the "Jizan" 08-23 rain event's own coordinates are ~74km
        # apart, and daily_precip_total differs hugely: 0.6mm at the city vs
        # 254.9mm at the storm centroid). Surfacing this distance lets the
        # agent/user understand why a same-named, same-day event can still
        # score a LOW similarity -- that is correct, hyperlocal behavior, not
        # a bug in this comparison.
        km = round(((city_lat - e_lat) ** 2 + (city_lon - e_lon) ** 2) ** 0.5 * 111)
        results.append({
            "event": e["label"], "event_date": e["date"], "event_location": e["location"],
            "event_headline_value": e["value"], "similarity_pct": similarity_pct,
            "features_compared": len(sq_dists),
            "event_distance_from_city_km": km,
        })

    results.sort(key=lambda r: -r["similarity_pct"])
    return {
        "city": city, "date": date, "hazard": hazard,
        "query_indicators": query_vec,
        "ranked_similar_events": results,
        "excluded_events": excluded,
        "note": ("similarity_pct is a descriptive, z-scored-distance-based measure "
                 "(100/(1+normalized_distance)), NOT a probability or a claim that this "
                 "event WILL repeat -- it only says how physically similar the raw "
                 "indicators are to a known past event. Each event's coordinates are its "
                 "own grid-cell MAXIMUM (the storm/heat centroid), often tens of km from a "
                 "same-named city's center -- a same-city, same-day comparison can still "
                 "score LOW if the extreme was hyperlocal (see event_distance_from_city_km "
                 "on each result); that is correct behavior, not an error."),
    }


# =============================================================================
# TOOL 5: region risk profile
# =============================================================================

# Hazards with a trained forecast model. "coastal" exists as a KG Hazard node
# (with a driven_by mechanism) but has no forecast model -- region_risk_tool
# must handle that gracefully (report KG exposure, skip forecast) rather than
# erroring or silently omitting the hazard.
FORECASTABLE_HAZARDS = {"flash_flood", "heatwave", "dust_storm"}


def region_risk_tool(city: str, date: str = None) -> dict:
    """Which hazards and mechanisms a city is exposed to, per the KG's
    at_risk_of/exposed_to edges (previously unused by any agent tool -- the
    other 3 tools are hazard-first, "why does flash_flood happen"; this one
    is city-first, "what should I worry about in this city"). If `date` is
    given, also attaches a live forecast_tool probability for each hazard
    that has a trained model.

    Args:
        city: one of CITIES keys
        date: optional "YYYY-MM-DD" -- if given, forecast probabilities are
            attached for the hazards that have a trained model
    """
    if city not in CITIES:
        return {"error": f"Unknown city '{city}'. Known cities: {list(CITIES.keys())}"}

    with open(KG_JSON, encoding="utf-8") as f:
        kg = json.load(f)
    node_by_id = {n["id"]: n for n in kg["nodes"]}

    hazards = [l["target"] for l in kg["links"]
              if l.get("etype") == "at_risk_of" and l["source"] == city]
    city_mechs = [l["target"] for l in kg["links"]
                 if l.get("etype") == "exposed_to" and l["source"] == city]

    if not hazards:
        return {"error": f"No at_risk_of data in the KG for '{city}' "
                         f"(this would indicate a real KG gap, not expected for any of the 8 known cities)"}

    profile = []
    date_error = None
    for hz in hazards:
        # full mechanism list for this hazard (driven_by), not just the
        # city-specific exposed_to subset -- exposed_to can be a partial
        # list (e.g. Jizan is exposed_to moisture_transport+orographic_lift,
        # both of flash_flood's 2 non-ARST mechanisms, but not ARST itself,
        # since ARST is a basin-wide not city-specific driver)
        hz_mechs = [l["target"] for l in kg["links"]
                   if l.get("etype") == "driven_by" and l["source"] == hz]
        entry = {
            "hazard": hz,
            "hazard_label": node_by_id.get(hz, {}).get("label", hz),
            "mechanisms_affecting_this_city": [m for m in city_mechs if m in hz_mechs],
            "all_mechanisms_for_this_hazard": hz_mechs,
            "has_forecast_model": hz in FORECASTABLE_HAZARDS,
        }
        if date is not None and hz in FORECASTABLE_HAZARDS:
            fr = forecast_tool(city, date, hz)
            if "error" in fr:
                date_error = fr["error"]   # same for every hazard (bad date/city), surface once
            else:
                entry["forecast"] = {"target_date": fr["target_date"], "probability": fr["probability"],
                                     "model_verified_roc_auc": fr["model_verified_roc_auc"]}
        profile.append(entry)

    result = {"city": city, "hazard_count": len(hazards), "hazards": profile}
    if date is not None:
        result["date"] = date
        if date_error:
            result["date_error"] = date_error   # e.g. out-of-range date -- disclosed, not silently dropped
    return result


# =============================================================================
# TOOL 6: CAP (Common Alerting Protocol) alert generator
# =============================================================================

import xml.etree.ElementTree as ET

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"

# Maps DetectionEngine's own percentile-derived severity labels (the same
# thresholds _reflexive_check cross-checks against -- see DETECTION_RULES
# above) onto the 4 CAP-standard severity values, rather than inventing a
# second, independent set of cutoffs.
CAP_SEVERITY_LABELS = {
    "low": "Minor", "medium": "Moderate", "high": "Severe", "extreme": "Extreme",
    "caution": "Minor", "warning": "Moderate", "alert": "Severe", "emergency": "Extreme",
}


def _cap_severity(hazard: str, probability: float):
    """probability -> (CAP severity string, underlying DetectionEngine label)."""
    levels = DETECTION_RULES[hazard]["severity"]
    rule_label = levels[0][0]
    for name, lo in levels:
        if probability >= lo:
            rule_label = name
    return CAP_SEVERITY_LABELS[rule_label], rule_label


def _build_cap_xml(identifier, sent, status, event, urgency, severity, certainty,
                    effective, expires, headline, description, instruction,
                    area_desc, lat, lon) -> str:
    """Serialize a CAP 1.2 <alert> via ElementTree (not string formatting) so
    all free-text fields (headline/description/instruction) are correctly
    XML-escaped."""
    ET.register_namespace("", CAP_NS)
    alert = ET.Element(f"{{{CAP_NS}}}alert")

    def sub(parent, tag, text):
        el = ET.SubElement(parent, f"{{{CAP_NS}}}{tag}")
        el.text = text
        return el

    sub(alert, "identifier", identifier)
    sub(alert, "sender", "mazu-demo@example.org")
    sub(alert, "sent", sent)
    sub(alert, "status", status)
    sub(alert, "msgType", "Alert")
    sub(alert, "scope", "Public")
    info = ET.SubElement(alert, f"{{{CAP_NS}}}info")
    sub(info, "category", "Met")
    sub(info, "event", event)
    sub(info, "urgency", urgency)
    sub(info, "severity", severity)
    sub(info, "certainty", certainty)
    sub(info, "effective", effective)
    sub(info, "expires", expires)
    sub(info, "senderName", "MAZU Saudi Multi-Hazard Early Warning (demo system)")
    sub(info, "headline", headline)
    sub(info, "description", description)
    sub(info, "instruction", instruction)
    area = ET.SubElement(info, f"{{{CAP_NS}}}area")
    sub(area, "areaDesc", area_desc)
    sub(area, "circle", f"{lat},{lon} 0")

    ET.indent(alert, space="  ")
    body = ET.tostring(alert, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body


def cap_alert_tool(city: str, target_date: str, hazard: str) -> dict:
    """Generate a CAP 1.2 (Common Alerting Protocol, OASIS international
    standard) XML alert from a forecast, so a warning can plug into real
    broadcast/siren/SMS infrastructure -- the same wire format MAZU's own
    national early-warning framework is built on. Internally calls
    forecast_tool, so severity/certainty are grounded in the exact same
    probability and reflexive-check output, not recomputed independently.

    Honesty note: `status` is hardcoded to "Exercise", never "Actual" -- this
    system runs on the 2025 historical dataset, not a live feed (see this
    module's top-of-file scope note), so a CAP message claiming "Actual"
    operational status would misrepresent the system.

    Args:
        city: one of CITIES keys
        target_date: "YYYY-MM-DD" -- the date being forecast
        hazard: "heatwave", "flash_flood", or "dust_storm"
    Returns dict with `cap_xml` (str) if the probability clears this hazard's
    lowest severity threshold, else `alert_warranted: False` (no alert issued
    for a low-probability day, matching real warning-system practice).
    """
    fr = forecast_tool(city, target_date, hazard)
    if "error" in fr:
        return fr

    probability = fr["probability"]
    cap_severity, rule_label = _cap_severity(hazard, probability)
    if rule_label in ("low", "caution"):
        threshold = DETECTION_RULES[hazard]["severity"][1][1]
        return {
            "alert_warranted": False,
            "city": city, "hazard": hazard, "target_date": fr["target_date"],
            "probability": probability,
            "reason": (f"Probability {probability:.2f} is below this hazard's lowest "
                       f"CAP-worthy threshold ({threshold:.2f}); no alert issued."),
        }

    with open(KG_JSON, encoding="utf-8") as f:
        kg = json.load(f)
    node_by_id = {n["id"]: n for n in kg["nodes"]}
    event_name = node_by_id.get(hazard, {}).get("label", hazard)

    rc = fr.get("reflexive_check") or {}
    # "Likely" only when the model AND the independent rule-based detection
    # engine both flag elevated risk (consistency == "consistent_elevated");
    # any disagreement, or a missing reflexive check, downgrades to "Possible"
    # -- CAP's own semantics ("Likely" = >50% confident) shouldn't be claimed
    # off a single, unconfirmed ML signal.
    certainty = "Likely" if rc.get("consistency") == "consistent_elevated" else "Possible"

    city_lat, city_lon = CITIES[city]
    identifier = f"MAZU-{hazard}-{city}-{fr['target_date']}"
    sent = f"{fr['features_from_date']}T18:00:00+03:00"   # issued end of the feature day, for target_date
    effective = f"{fr['target_date']}T00:00:00+03:00"
    expires = f"{fr['target_date']}T23:59:59+03:00"

    headline = f"{event_name} risk for {city}, Saudi Arabia on {fr['target_date']}"
    description = (
        f"MAZU multi-hazard early warning system forecasts a {probability:.0%} probability "
        f"of {event_name.lower()} affecting {city} on {fr['target_date']}, based on "
        f"indicators from {fr['features_from_date']}. Model verified ROC-AUC: "
        f"{fr['model_verified_roc_auc']:.3f}. Independent rule-based cross-check: "
        f"{rc.get('consistency', 'unavailable')}."
    )
    instruction = (
        fr["terrain_note"] if fr.get("terrain_note")
        else "Follow guidance from the Saudi National Center of Meteorology (NCM) and local authorities."
    )

    cap_xml = _build_cap_xml(
        identifier=identifier, sent=sent, status="Exercise", event=event_name,
        urgency="Expected", severity=cap_severity, certainty=certainty,
        effective=effective, expires=expires, headline=headline,
        description=description, instruction=instruction,
        area_desc=f"{city}, Saudi Arabia", lat=round(city_lat, 2), lon=round(city_lon, 2),
    )

    return {
        "alert_warranted": True,
        "city": city, "hazard": hazard, "target_date": fr["target_date"],
        "probability": probability, "cap_severity": cap_severity, "cap_certainty": certainty,
        "cap_xml": cap_xml,
    }


if __name__ == "__main__":
    print("tools.py loaded OK — run test_tools.py for verification.")
