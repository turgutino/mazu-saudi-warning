# =============================================================================
# Retrain flash_flood / heatwave / dust_storm using the EXACT same methodology
# as model/03_forecast_baseline.py and model/07_dust_storm_forecast.py (same
# HistGradientBoostingClassifier params, same stride-2 grid-cell-day sampling,
# same Jan-Jun/Jul-Dec time-based split, same POD/FAR/CSI metrics), once
# WITHOUT and once WITH the new gc_* (GraphCast-distilled) features, and
# prints a side-by-side comparison.
#
# Runs LOCALLY. Requires experiments/graphcast_distillation/mazu_dataset_with_graphcast.nc
# to already exist (produced by 03_integrate_feature.py).
#
# Saves candidate model files under models_candidate/ -- NEVER touches
# model/saved_models/*.joblib. Deployment (if warranted) is a separate,
# explicit, user-approved step.
# =============================================================================
import importlib.util
import os

import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "mazu_dataset_with_graphcast.nc")
MODEL_DIR = os.path.join(HERE, "..", "..", "model")
CANDIDATE_DIR = os.path.join(HERE, "models_candidate")
os.makedirs(CANDIDATE_DIR, exist_ok=True)

BASE_FEATURE_VARS = [
    "daily_precip_total", "daily_convective_precip", "daily_large_scale_precip",
    "t2m_c", "tmax_c", "tmin_c", "heat_index_c", "vpd_kpa",
    "cape", "pwat", "ivt", "wind850_speed", "wind_shear_850_200",
    "daily_precip_anomaly", "t2m_anomaly_c", "tmax_anomaly_c", "sst_celsius",
]
DUST_EXTRA_VARS = ["wind10_speed", "dewpoint_depression_c"]
GC_FEATURE_VARS = ["gc_t2m", "gc_u10m", "gc_v10m", "gc_msl", "gc_precip"]

TRAIN_END = "2025-06-30"
STRIDE = 2

# same rule-engine dust-storm label construction as 07_dust_storm_forecast.py
_de_spec = importlib.util.spec_from_file_location("de", os.path.join(MODEL_DIR, "01_detection_engine.py"))
_de = importlib.util.module_from_spec(_de_spec)
_de_src = open(os.path.join(MODEL_DIR, "01_detection_engine.py"), encoding="utf-8").read()
_de_src = _de_src.split('if __name__ == "__main__":')[0]
exec(compile(_de_src, "de", "exec"), _de.__dict__)
DUST_RULE = _de.RULES["dust_storm"]


def build_dust_label(ds):
    n_t, n_lat, n_lon = ds.dims["time"], ds.dims["latitude"], ds.dims["longitude"]
    score = np.zeros((n_t, n_lat, n_lon), dtype="float32")
    wsum = np.zeros_like(score)
    ops = {">=": np.greater_equal, ">": np.greater, "<=": np.less_equal, "<": np.less}
    for c in DUST_RULE["conditions"]:
        a = ds[c["ind"]].values
        valid = np.isfinite(a)
        hit = np.zeros_like(score)
        hit[valid] = ops[c["op"]](a[valid], c["thr"]).astype("float32")
        score += hit * c["w"]
        wsum += valid.astype("float32") * c["w"]
    risk = np.where(wsum > 0, score / wsum, 0.0)
    return (risk >= DUST_RULE["risk_threshold"]).astype("int8")


TARGETS = {
    "flash_flood": {"label_var": "flash_flood_risk", "label_thr": 2, "extra_vars": []},
    "heatwave":    {"label_var": "heatwave_day_flag", "label_thr": 1, "extra_vars": []},
    "dust_storm":  {"label_var": None, "label_thr": None, "extra_vars": DUST_EXTRA_VARS},
}


def build_supervised(ds, hazard, use_graphcast, dust_label_all=None):
    times = np.array([str(t)[:10] for t in ds.time.values])
    n_t, n_lat, n_lon = len(times), len(ds.latitude), len(ds.longitude)
    lat, lon = ds.latitude.values, ds.longitude.values
    doy = ds.time.values.astype("datetime64[D]").astype(object)
    doy = np.array([d.timetuple().tm_yday for d in doy])

    cfg = TARGETS[hazard]
    if hazard == "dust_storm":
        label_all = dust_label_all
    else:
        label_all = ds[cfg["label_var"]].values

    yi = np.arange(0, n_lat, STRIDE)
    xi = np.arange(0, n_lon, STRIDE)
    LAT2, LON2 = np.meshgrid(lat[yi], lon[xi], indexing="ij")
    lat_flat, lon_flat = LAT2.ravel(), LON2.ravel()
    n_cells = lat_flat.size

    feature_vars = BASE_FEATURE_VARS + cfg["extra_vars"]
    if use_graphcast:
        feature_vars = feature_vars + GC_FEATURE_VARS

    feat_stack = np.stack([ds[v].values[:, yi][:, :, xi] for v in feature_vars], axis=-1)

    rows_X, rows_y, rows_date = [], [], []
    for ti in range(n_t - 1):
        X_t = feat_stack[ti].reshape(n_cells, -1)
        if hazard == "dust_storm":
            y_next = label_all[ti + 1][yi][:, xi].reshape(n_cells).astype("int8")
        else:
            y_next = (label_all[ti + 1][yi][:, xi].reshape(n_cells) >= cfg["label_thr"]).astype("int8")
        valid = np.isfinite(y_next)
        if valid.sum() == 0:
            continue
        extra = np.column_stack([lat_flat, lon_flat, np.full(n_cells, doy[ti])])
        Xrow = np.column_stack([X_t, extra])[valid]
        rows_X.append(Xrow)
        rows_y.append(y_next[valid])
        rows_date.append(np.full(valid.sum(), times[ti + 1]))

    X = np.concatenate(rows_X, axis=0)
    y = np.concatenate(rows_y, axis=0)
    dates = np.concatenate(rows_date, axis=0)
    return X, y, dates


def train_and_eval(ds, hazard, use_graphcast, dust_label_all=None):
    X, y, dates = build_supervised(ds, hazard, use_graphcast, dust_label_all)
    train_mask = dates <= TRAIN_END
    test_mask = ~train_mask
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]

    clf = HistGradientBoostingClassifier(
        max_iter=150, max_depth=6, learning_rate=0.08,
        class_weight="balanced", random_state=42, early_stopping=True,
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    pod = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (tp + fp) if (tp + fp) else 0.0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) else 0.0
    roc = roc_auc_score(yte, proba) if len(np.unique(yte)) > 1 else float("nan")
    pr_auc = average_precision_score(yte, proba) if len(np.unique(yte)) > 1 else float("nan")

    return clf, {
        "n_train": len(ytr), "n_test": len(yte), "pos_rate_test": float(yte.mean()),
        "ROC_AUC": roc, "PR_AUC": pr_auc, "POD": pod, "FAR": far, "CSI": csi,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def main():
    if not os.path.exists(DATASET):
        raise SystemExit(
            f"{DATASET} not found. Run 03_integrate_feature.py first "
            "(needs raw_graphcast_daily/ populated from the server run)."
        )

    ds = xr.open_dataset(DATASET)
    dust_label_all = build_dust_label(ds)

    print("=" * 78)
    print("BASELINE vs. GRAPHCAST-DISTILLED FEATURE — side-by-side comparison")
    print("=" * 78)

    results = {}
    for hazard in TARGETS:
        print(f"\n### {hazard} ###")
        for use_gc in (False, True):
            label = "WITH graphcast" if use_gc else "baseline (no graphcast)"
            clf, m = train_and_eval(ds, hazard, use_gc, dust_label_all)
            results[(hazard, use_gc)] = m
            print(f"  [{label}] ROC-AUC={m['ROC_AUC']:.4f} PR-AUC={m['PR_AUC']:.4f} "
                  f"POD={m['POD']:.4f} FAR={m['FAR']:.4f} CSI={m['CSI']:.4f} "
                  f"(train={m['n_train']:,} test={m['n_test']:,})")
            if use_gc:
                import joblib
                joblib.dump(clf, os.path.join(CANDIDATE_DIR, f"{hazard}_model_with_graphcast.joblib"))

        base = results[(hazard, False)]
        gc = results[(hazard, True)]
        d_pod = gc["POD"] - base["POD"]
        d_far = gc["FAR"] - base["FAR"]
        d_csi = gc["CSI"] - base["CSI"]
        d_roc = gc["ROC_AUC"] - base["ROC_AUC"]
        verdict = "IMPROVED" if (d_pod > 0.01 and d_csi >= -0.005) else "NO CLEAN WIN"
        print(f"  DELTA: ROC-AUC={d_roc:+.4f} POD={d_pod:+.4f} FAR={d_far:+.4f} CSI={d_csi:+.4f}  -> {verdict}")

    ds.close()
    print(f"\nCandidate models saved to {CANDIDATE_DIR} (model/saved_models/ untouched).")


if __name__ == "__main__":
    main()
