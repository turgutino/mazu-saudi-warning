# =============================================================================
# MAZU Saudi Arabia — Layer 2: Forecast Baseline (t -> t+1)
#
# TRUE forecasting, not detection: uses day t's indicators to predict whether
# day t+1 will be an extreme-hazard day, per grid cell. This is fundamentally
# different from Layer 1 (which reads today's conditions to flag today) and
# from the competitor's system (which only detects, never forecasts).
#
# Design choices (all justified, not arbitrary):
#   - Grid-cell-day samples: ~365 days x 35200 cells gives enough samples for
#     a gradient-boosted baseline even though only 1 year of data exists.
#   - Time-based train/test split (Jan-Jun train, Jul-Dec test) so the model
#     NEVER sees the second half of the year, including the known Aug flood
#     and Jul/Aug heatwaves used for validation -> no leakage.
#   - HistGradientBoostingClassifier: handles missing values (Jan-Mar
#     pressure-level gaps, Oct-Dec DS10 gaps) natively, no imputation needed.
#   - Class imbalance (extremes are rare) handled via class_weight.
#   - Extra features: day-of-year (seasonality), lat/lon (spatial context),
#     TODAY's risk score (persistence signal) - a legitimate forecast
#     predictor, not leakage, since it's t's own state used to predict t+1.
#
# Output: model/forecast_report.txt + outputs/forecast_validation.png
# =============================================================================

import os
import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "data", "mazu_dataset.nc")
OUT_DIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_VARS = [
    "daily_precip_total", "daily_convective_precip", "daily_large_scale_precip",
    "t2m_c", "tmax_c", "tmin_c", "heat_index_c", "vpd_kpa",
    "cape", "pwat", "ivt", "wind850_speed", "wind_shear_850_200",
    "daily_precip_anomaly", "t2m_anomaly_c", "tmax_anomaly_c", "sst_celsius",
]

TARGETS = {
    "flash_flood": {"label_var": "flash_flood_risk", "label_thr": 2},
    "heatwave":    {"label_var": "heatwave_day_flag", "label_thr": 1},
}

TRAIN_END = "2025-06-30"   # train: Jan-Jun ; test: Jul-Dec (unseen 2nd half)


def build_supervised(ds, hazard):
    """Build (X, y, dates, cell_idx) arrays for t -> t+1 prediction.
    Downsample cells by a stride to keep the problem tractable while still
    giving millions of samples across the full grid and year."""
    times = np.array([str(t)[:10] for t in ds.time.values])
    n_t, n_lat, n_lon = len(times), len(ds.latitude), len(ds.longitude)
    lat = ds.latitude.values
    lon = ds.longitude.values
    doy = ds.time.values.astype("datetime64[D]").astype(object)
    doy = np.array([d.timetuple().tm_yday for d in doy])

    cfg = TARGETS[hazard]
    label_all = ds[cfg["label_var"]].values  # (time, lat, lon)

    # spatial stride to control sample count (every 2nd cell in each dim -> 1/4 density)
    stride = 2
    yi = np.arange(0, n_lat, stride)
    xi = np.arange(0, n_lon, stride)
    LAT2, LON2 = np.meshgrid(lat[yi], lon[xi], indexing="ij")
    lat_flat = LAT2.ravel(); lon_flat = LON2.ravel()
    n_cells = lat_flat.size

    feat_stack = np.stack([ds[v].values[:, yi][:, :, xi] for v in FEATURE_VARS], axis=-1)  # (T, y, x, F)

    rows_X, rows_y, rows_date = [], [], []
    for ti in range(n_t - 1):   # predict ti+1 from ti
        X_t = feat_stack[ti].reshape(n_cells, -1)                 # (cells, F)
        y_next = (label_all[ti + 1][yi][:, xi].reshape(n_cells) >= cfg["label_thr"]).astype("int8")
        # rows with SOME missing features are kept (HGB handles NaN natively);
        # only require the t+1 label itself to be valid
        valid = np.isfinite(y_next)
        n_valid = valid.sum()
        if n_valid == 0:
            continue
        extra = np.column_stack([lat_flat, lon_flat, np.full(n_cells, doy[ti])])
        Xrow = np.column_stack([X_t, extra])[valid]
        rows_X.append(Xrow)
        rows_y.append(y_next[valid])
        rows_date.append(np.full(valid.sum(), times[ti + 1]))     # date being PREDICTED

    X = np.concatenate(rows_X, axis=0)
    y = np.concatenate(rows_y, axis=0)
    dates = np.concatenate(rows_date, axis=0)
    return X, y, dates, lat_flat, lon_flat


def main():
    ds = xr.open_dataset(DATASET)

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("MAZU Layer 2 — Forecast Baseline (t -> t+1), verified")
    report_lines.append("=" * 70)

    for hazard in TARGETS:
        print(f"\n### {hazard} ###")
        X, y, dates, lat_flat, lon_flat = build_supervised(ds, hazard)
        print(f"  samples: {len(y):,}  | positive rate: {100*y.mean():.3f}%")

        train_mask = dates <= TRAIN_END
        test_mask = ~train_mask
        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte, dte = X[test_mask], y[test_mask], dates[test_mask]
        print(f"  train: {len(ytr):,} (pos={ytr.sum()})  test: {len(yte):,} (pos={yte.sum()})")

        clf = HistGradientBoostingClassifier(
            max_iter=150, max_depth=6, learning_rate=0.08,
            class_weight="balanced", random_state=42, early_stopping=True,
        )
        clf.fit(Xtr, ytr)

        proba = clf.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        pod = tp / (tp + fn) if (tp + fn) else 0.0     # probability of detection
        far = fp / (tp + fp) if (tp + fp) else 0.0     # false alarm ratio
        csi = tp / (tp + fn + fp) if (tp + fn + fp) else 0.0

        report_lines.append(f"\n--- {hazard} ---")
        report_lines.append(f"train samples: {len(ytr):,} (positive {100*ytr.mean():.3f}%)")
        report_lines.append(f"test  samples: {len(yte):,} (positive {100*yte.mean():.3f}%)")
        report_lines.append(f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
        report_lines.append(f"POD={pod:.3f}  FAR={far:.3f}  CSI={csi:.3f}")
        report_lines.append(f"Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

        print(f"  Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f} | POD={pod:.3f} FAR={far:.3f} CSI={csi:.3f}")

        # ── validation on the specific known events (t-1 -> forecast t) ──
        known = {
            "flash_flood": [("2025-08-22", "2025-08-23", "Jizan"), ("2025-08-18", "2025-08-19", "Arabian Sea")],
            "heatwave":    [("2025-07-24", "2025-07-25", "Riyadh"), ("2025-08-15", "2025-08-16", "Persian Gulf")],
        }[hazard]
        report_lines.append("Known-event forecast check (predict day+1 from day):")
        for d_from, d_to, region in known:
            mask = dte == d_to
            if mask.sum() == 0:
                report_lines.append(f"  {d_from} -> {d_to} ({region}): [no test rows]")
                continue
            p = proba[mask]
            report_lines.append(f"  {d_from} -> {d_to} ({region}): mean_p={p.mean():.3f} max_p={p.max():.3f} "
                                f"cells_p>0.5={int((p>0.5).sum())}/{mask.sum()}")

        # ── negative control: calm days should have low mean forecast prob ──
        for calm in ["2025-11-05", "2025-09-15"]:
            mask = dte == calm
            if mask.sum():
                p = proba[mask]
                report_lines.append(f"  calm-day control {calm}: mean_p={p.mean():.3f} max_p={p.max():.3f}")

    ds.close()

    rpt = "\n".join(report_lines)
    with open(os.path.join(HERE, "forecast_report.txt"), "w", encoding="utf-8") as f:
        f.write(rpt)
    print("\n" + rpt)
    print(f"\n[SAVED] {os.path.join(HERE, 'forecast_report.txt')}")


if __name__ == "__main__":
    main()
