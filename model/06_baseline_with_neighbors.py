# =============================================================================
# MAZU — Option A: cheap spatial-context feature engineered into the Layer-2a
# baseline (HistGradientBoosting), instead of a full deep GNN.
#
# Hypothesis (from Layer 2b's honest finding): explicit spatial context genuinely
# helped the GNN find the Aug-23 flash-flood hotspot, but full message passing
# also hurt calibration everywhere else. This tests whether we can capture the
# useful part cheaply: add, for each cell, the 4-connected-neighbour MEAN of the
# key indicators at day t as extra columns, then retrain the SAME tree model on
# the SAME train/test split used in Layer 2a, for a direct, fair comparison.
# =============================================================================

import os
import importlib.util
import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fb", os.path.join(HERE, "03_forecast_baseline.py"))
fb = importlib.util.module_from_spec(spec)
src = open(os.path.join(HERE, "03_forecast_baseline.py"), encoding="utf-8").read()
src = src.replace('if __name__ == "__main__":\n    main()', "")
exec(compile(src, "fb", "exec"), fb.__dict__)

# Neighbour-mean is computed for the physically dominant drivers only (from the
# Layer-2a permutation-importance ranking), not all 16 features, to keep the
# added dimensionality modest and interpretable.
NEIGHBOR_VARS = ["cape", "daily_precip_total", "ivt", "vpd_kpa", "wind_shear_850_200",
                 "pwat", "daily_convective_precip"]


def neighbor_mean(arr):
    """arr: (n_lat, n_lon) -> mean of the up/down/left/right neighbours (self excluded),
    with correct edge handling (average over however many neighbours exist)."""
    n_lat, n_lon = arr.shape
    total = np.zeros_like(arr, dtype="float64")
    count = np.zeros_like(arr, dtype="float64")
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)

    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in shifts:
        shifted_val = np.zeros_like(arr, dtype="float64")
        shifted_valid = np.zeros_like(arr, dtype=bool)
        ys, ye = max(0, dy), n_lat + min(0, dy)
        xs, xe = max(0, dx), n_lon + min(0, dx)
        ys2, ye2 = max(0, -dy), n_lat + min(0, -dy)
        xs2, xe2 = max(0, -dx), n_lon + min(0, -dx)
        shifted_val[ys2:ye2, xs2:xe2] = filled[ys:ye, xs:xe]
        shifted_valid[ys2:ye2, xs2:xe2] = valid[ys:ye, xs:xe]
        total += np.where(shifted_valid, shifted_val, 0.0)
        count += shifted_valid.astype("float64")

    out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out.astype("float32")


def build_supervised_with_neighbors(ds, hazard):
    times = np.array([str(t)[:10] for t in ds.time.values])
    n_t = len(times)
    lat, lon = ds.latitude.values, ds.longitude.values
    stride = 2
    yi = np.arange(0, len(lat), stride)
    xi = np.arange(0, len(lon), stride)
    n_lat_s, n_lon_s = len(yi), len(xi)
    LAT2, LON2 = np.meshgrid(lat[yi], lon[xi], indexing="ij")
    lat_flat, lon_flat = LAT2.ravel(), LON2.ravel()
    n_cells = lat_flat.size
    doy = np.array([d.timetuple().tm_yday for d in
                    ds.time.values.astype("datetime64[D]").astype(object)])

    cfg = fb.TARGETS[hazard]
    label_all = ds[cfg["label_var"]].values
    feat_stack = np.stack([ds[v].values[:, yi][:, :, xi] for v in fb.FEATURE_VARS], axis=-1)  # (T,y,x,F)
    neigh_idx = [fb.FEATURE_VARS.index(v) for v in NEIGHBOR_VARS]

    rows_X, rows_y, rows_date = [], [], []
    for ti in range(n_t - 1):
        X_t = feat_stack[ti].reshape(n_cells, -1)
        neigh_cols = np.stack([neighbor_mean(feat_stack[ti, :, :, k]) for k in neigh_idx], axis=-1)
        neigh_cols = neigh_cols.reshape(n_cells, -1)
        y_next = (label_all[ti + 1][yi][:, xi].reshape(n_cells) >= cfg["label_thr"]).astype("int8")
        valid = np.isfinite(y_next)
        extra = np.column_stack([lat_flat, lon_flat, np.full(n_cells, doy[ti])])
        Xrow = np.column_stack([X_t, neigh_cols, extra])[valid]
        rows_X.append(Xrow)
        rows_y.append(y_next[valid])
        rows_date.append(np.full(valid.sum(), times[ti + 1]))

    X = np.concatenate(rows_X, axis=0)
    y = np.concatenate(rows_y, axis=0)
    dates = np.concatenate(rows_date, axis=0)
    return X, y, dates


def main():
    ds = xr.open_dataset(fb.DATASET)
    lines = ["=" * 70, "MAZU — Option A: neighbour-mean spatial-context feature", "=" * 70,
             f"Added features: neighbour-mean of {NEIGHBOR_VARS}",
             "Same model (HistGradientBoosting), same train/test split as Layer 2a.", ""]

    known = {
        "flash_flood": [("2025-08-22", "2025-08-23", "Jizan 254.9mm"), ("2025-08-18", "2025-08-19", "Arabian Sea IVT 728")],
        "heatwave":    [("2025-07-24", "2025-07-25", "Riyadh Tmax 53.7C"), ("2025-08-15", "2025-08-16", "Persian Gulf heat-index")],
    }
    baseline_scores = {"heatwave": {"roc": 0.958, "prauc": 0.756}, "flash_flood": {"roc": 0.873, "prauc": 0.089}}

    summary = {}
    for hazard in ["heatwave", "flash_flood"]:
        print(f"\n### {hazard} ###")
        X, y, dates = build_supervised_with_neighbors(ds, hazard)
        train_mask = dates <= fb.TRAIN_END
        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte, dte = X[~train_mask], y[~train_mask], dates[~train_mask]
        print(f"  train: {len(ytr):,} (pos={ytr.sum()})  test: {len(yte):,} (pos={yte.sum()})")

        clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                             class_weight="balanced", random_state=42, early_stopping=True)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]

        roc = roc_auc_score(yte, proba)
        prauc = average_precision_score(yte, proba)
        b = baseline_scores[hazard]
        print(f"  ROC-AUC={roc:.3f} (baseline {b['roc']:.3f}, delta {roc-b['roc']:+.3f})")
        print(f"  PR-AUC ={prauc:.3f} (baseline {b['prauc']:.3f}, delta {prauc-b['prauc']:+.3f})")

        lines.append(f"--- {hazard} ---")
        lines.append(f"train: {len(ytr):,} (pos={ytr.sum()})  test: {len(yte):,} (pos={yte.sum()})")
        lines.append(f"ROC-AUC = {roc:.3f}   (Layer-2a baseline {b['roc']:.3f}, delta {roc-b['roc']:+.3f})")
        lines.append(f"PR-AUC  = {prauc:.3f}   (Layer-2a baseline {b['prauc']:.3f}, delta {prauc-b['prauc']:+.3f})")

        for d_from, d_to, note in known[hazard]:
            mask = dte == d_to
            if mask.sum():
                p = proba[mask]
                lines.append(f"  {d_from}->{d_to} ({note}): mean_p={p.mean():.3f} max_p={p.max():.3f} "
                            f"cells_p>0.5={int((p>0.5).sum())}/{mask.sum()}")
        for calm in ["2025-11-05"]:
            mask = dte == calm
            if mask.sum():
                p = proba[mask]
                lines.append(f"  calm control {calm}: mean_p={p.mean():.3f} max_p={p.max():.3f}")
        lines.append("")
        summary[hazard] = {"roc": roc, "prauc": prauc}

    lines.append("=" * 70)
    lines.append("VERDICT")
    lines.append("=" * 70)
    for hz in ["heatwave", "flash_flood"]:
        b = baseline_scores[hz]; s = summary[hz]
        verdict = "IMPROVED" if (s["roc"] >= b["roc"] and s["prauc"] >= b["prauc"]) else \
                  "WORSE" if (s["roc"] <= b["roc"] and s["prauc"] <= b["prauc"]) else "MIXED"
        lines.append(f"{hz}: {verdict}  (ROC {b['roc']:.3f}->{s['roc']:.3f}, PR-AUC {b['prauc']:.3f}->{s['prauc']:.3f})")

    rpt = "\n".join(lines)
    with open(os.path.join(HERE, "neighbor_feature_report.txt"), "w", encoding="utf-8") as f:
        f.write(rpt)
    print("\n" + rpt)
    ds.close()


if __name__ == "__main__":
    main()
