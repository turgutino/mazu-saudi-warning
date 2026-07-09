# =============================================================================
# MAZU Saudi Arabia — dust_storm forecast (t -> t+1), matching Layer 2's
# verified t->t+1 methodology exactly: same HistGradientBoostingClassifier,
# same time-based train/test split (Jan-Jun train, Jul-Dec test, no leakage),
# same rare-event metrics (ROC-AUC/PR-AUC, not accuracy).
#
# Label: no pre-computed "dust_storm" flag exists in the dataset (unlike
# flash_flood_risk / heatwave_day_flag, which were teacher-provided), so the
# label is built from Layer 1's own already-validated, data-grounded RULES
# threshold (01_detection_engine.py's dust_storm rule, risk_threshold=0.55)
# -- reusing an existing, tested definition rather than inventing a new one.
# =============================================================================

import os
import sys
import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "data", "mazu_dataset.nc")

sys.path.insert(0, HERE)
import importlib.util
_de_spec = importlib.util.spec_from_file_location("de", os.path.join(HERE, "01_detection_engine.py"))
_de = importlib.util.module_from_spec(_de_spec)
_de_src = open(os.path.join(HERE, "01_detection_engine.py"), encoding="utf-8").read()
_de_src = _de_src.split('if __name__ == "__main__":')[0]
exec(compile(_de_src, "de", "exec"), _de.__dict__)
DUST_RULE = _de.RULES["dust_storm"]

FEATURE_VARS = [
    "daily_precip_total", "daily_convective_precip", "daily_large_scale_precip",
    "t2m_c", "tmax_c", "tmin_c", "heat_index_c", "vpd_kpa",
    "cape", "pwat", "ivt", "wind850_speed", "wind_shear_850_200",
    "daily_precip_anomaly", "t2m_anomaly_c", "tmax_anomaly_c", "sst_celsius",
    "wind10_speed", "dewpoint_depression_c",
]

TRAIN_END = "2025-06-30"


def build_dust_label(ds):
    """Vectorized re-implementation of DetectionEngine.risk_field for
    dust_storm across the WHOLE time axis at once (the class-based version
    computes one day at a time, which would mean 365 slow repeated calls
    here) -- same weights/thresholds as 01_detection_engine.py's RULES, kept
    in sync by reading DUST_RULE directly rather than hardcoding a second
    copy of the numbers."""
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


def build_supervised(ds, label_all):
    times = np.array([str(t)[:10] for t in ds.time.values])
    n_t, n_lat, n_lon = len(times), len(ds.latitude), len(ds.longitude)
    lat = ds.latitude.values
    lon = ds.longitude.values
    doy = ds.time.values.astype("datetime64[D]").astype(object)
    doy = np.array([d.timetuple().tm_yday for d in doy])

    stride = 2
    yi = np.arange(0, n_lat, stride)
    xi = np.arange(0, n_lon, stride)
    LAT2, LON2 = np.meshgrid(lat[yi], lon[xi], indexing="ij")
    lat_flat = LAT2.ravel(); lon_flat = LON2.ravel()
    n_cells = lat_flat.size

    feat_stack = np.stack([ds[v].values[:, yi][:, :, xi] for v in FEATURE_VARS], axis=-1)

    rows_X, rows_y, rows_date = [], [], []
    for ti in range(n_t - 1):
        X_t = feat_stack[ti].reshape(n_cells, -1)
        y_next = label_all[ti + 1][yi][:, xi].reshape(n_cells)
        valid = np.isfinite(y_next.astype("float32"))
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
    return X, y, dates, lat_flat, lon_flat


def main():
    ds = xr.open_dataset(DATASET)
    print("Building dust_storm label from Layer 1's validated RULES threshold...")
    label_all = build_dust_label(ds)
    print(f"  overall positive rate: {100*np.nanmean(label_all):.3f}%")

    X, y, dates, lat_flat, lon_flat = build_supervised(ds, label_all)
    print(f"samples: {len(y):,} | positive rate: {100*y.mean():.3f}%")

    train_mask = dates <= TRAIN_END
    test_mask = ~train_mask
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte, dte = X[test_mask], y[test_mask], dates[test_mask]
    print(f"train: {len(ytr):,} (pos={ytr.sum()})  test: {len(yte):,} (pos={yte.sum()})")

    clf = HistGradientBoostingClassifier(
        max_iter=150, max_depth=6, learning_rate=0.08,
        class_weight="balanced", random_state=42, early_stopping=True,
    )
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    roc_auc = roc_auc_score(yte, proba)
    pr_auc = average_precision_score(yte, proba)
    pred = (proba >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("MAZU — dust_storm forecast baseline (t -> t+1), verified")
    report_lines.append("=" * 70)
    report_lines.append(f"Label source: Layer 1's DetectionEngine RULES['dust_storm'] "
                        f"(risk_threshold={DUST_RULE['risk_threshold']}), not a separate invented definition")
    report_lines.append(f"train samples: {len(ytr):,} (positive {100*ytr.mean():.3f}%)")
    report_lines.append(f"test  samples: {len(yte):,} (positive {100*yte.mean():.3f}%)")
    report_lines.append(f"ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    report_lines.append(f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    report_lines.append(f"Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    print(f"ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

    # ── known-event validation: predict 07-06 from 07-05's indicators ──
    # (07-06, not the 06-19 event used for detection validation, because
    # 06-19 falls in the TRAINING period (<=06-30) -- testing forecast skill
    # requires an event the model never saw, found the same way: largest
    # dust-rule cluster count within the Jul-Dec test window.)
    report_lines.append("\nKnown-event forecast check (predict day+1 from day):")
    for d_from, d_to, region in [("2025-07-05", "2025-07-06", "widespread, largest in test period")]:
        mask = dte == d_to
        if mask.sum():
            p = proba[mask]
            report_lines.append(f"  {d_from} -> {d_to} ({region}): mean_p={p.mean():.3f} max_p={p.max():.3f} "
                                f"cells_p>0.5={int((p>0.5).sum())}/{mask.sum()}")

    # ── negative controls ──
    for calm in ["2025-11-05", "2025-09-15"]:
        mask = dte == calm
        if mask.sum():
            p = proba[mask]
            report_lines.append(f"  calm-day control {calm}: mean_p={p.mean():.3f} max_p={p.max():.3f}")

    ds.close()

    rpt = "\n".join(report_lines)
    with open(os.path.join(HERE, "dust_storm_forecast_report.txt"), "w", encoding="utf-8") as f:
        f.write(rpt)
    print("\n" + rpt)

    # ── save the production model (for agent tool use) ──
    import joblib
    SAVED_DIR = os.path.join(HERE, "..", "agent", "saved_models")
    os.makedirs(SAVED_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(SAVED_DIR, "dust_storm_model.joblib"))
    print(f"\n[SAVED] {os.path.join(SAVED_DIR, 'dust_storm_model.joblib')}")
    return roc_auc, pr_auc


if __name__ == "__main__":
    main()
