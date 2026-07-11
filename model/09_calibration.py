# =============================================================================
# MAZU — reliability diagrams (calibration curves) for all 3 hazards
#
# Answers a question POD/FAR/CSI/HSS (single fixed threshold) and ROC-AUC/
# PR-AUC (threshold-independent RANKING quality) do NOT answer: "when the
# model says 70%, does the event actually happen ~70% of the time?" -- i.e.
# is the model's probability itself trustworthy, not just its ranking.
#
# Uses the ALREADY-SAVED, already-verified production models (joblib) --
# does NOT retrain -- on the exact same held-out Jul-Dec test set as
# 08_meteorological_metrics.py, so these numbers cannot drift from what the
# agent actually serves.
# =============================================================================

import os
import sys
import json
import importlib.util
import joblib
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(HERE, "..", "agent", "saved_models")
OUT_DIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

N_BINS = 10  # standard reliability-diagram bin count (Wilks, "Statistical
             # Methods in the Atmospheric Sciences"), equal-width [0,1] bins


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reliability_bins(y_true, y_proba, n_bins=N_BINS):
    """Standard equal-width reliability-diagram binning. Returns per-bin
    dicts (even for empty bins -- disclosed as count=0, not silently
    dropped) plus overall Brier score and Expected Calibration Error (ECE)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin is closed on both ends so p==1.0 is included
        if i == n_bins - 1:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)
        count = int(mask.sum())
        if count > 0:
            mean_pred = float(y_proba[mask].mean())
            obs_freq = float(y_true[mask].mean())
            ece += (count / n) * abs(mean_pred - obs_freq)
        else:
            mean_pred, obs_freq = None, None
        bins.append({"bin_lo": round(float(lo), 2), "bin_hi": round(float(hi), 2),
                     "count": count, "mean_predicted": mean_pred, "observed_freq": obs_freq})
    brier = float(np.mean((y_proba - y_true) ** 2))
    return bins, round(ece, 4), round(brier, 4)


def main():
    de = load_module("de_c", os.path.join(HERE, "01_detection_engine.py"))
    fb = load_module("fb_c", os.path.join(HERE, "03_forecast_baseline.py"))
    bn = load_module("bn_c", os.path.join(HERE, "06_baseline_with_neighbors.py"))
    dust = load_module("dust_c", os.path.join(HERE, "07_dust_storm_forecast.py"))

    ds = xr.open_dataset(fb.DATASET)
    results = {}

    # heatwave
    X, y, dates = bn.build_supervised_with_neighbors(ds, "heatwave")
    te = dates > fb.TRAIN_END
    clf = joblib.load(os.path.join(SAVED_DIR, "heatwave_model.joblib"))
    proba = clf.predict_proba(X[te])[:, 1]
    results["heatwave"] = (y[te], proba)

    # flash_flood
    X2, y2, dates2, _, _ = fb.build_supervised(ds, "flash_flood")
    te2 = dates2 > fb.TRAIN_END
    clf2 = joblib.load(os.path.join(SAVED_DIR, "flash_flood_model.joblib"))
    proba2 = clf2.predict_proba(X2[te2])[:, 1]
    results["flash_flood"] = (y2[te2], proba2)

    # dust_storm
    label_all = dust.build_dust_label(ds)
    X3, y3, dates3, _, _ = dust.build_supervised(ds, label_all)
    te3 = dates3 > dust.TRAIN_END
    clf3 = joblib.load(os.path.join(SAVED_DIR, "dust_storm_model.joblib"))
    proba3 = clf3.predict_proba(X3[te3])[:, 1]
    results["dust_storm"] = (y3[te3], proba3)

    ds.close()

    report = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"heatwave": "#d99a2b", "flash_flood": "#065A82", "dust_storm": "#8a5a2b"}
    for ax, hz in zip(axes, ("heatwave", "flash_flood", "dust_storm")):
        y_true, y_proba = results[hz]
        bins, ece, brier = reliability_bins(y_true, y_proba)
        report[hz] = {"n_bins": N_BINS, "ece": ece, "brier_score": brier,
                      "n_test_samples": int(len(y_true)), "bins": bins}

        xs = [b["mean_predicted"] for b in bins if b["mean_predicted"] is not None]
        ys = [b["observed_freq"] for b in bins if b["observed_freq"] is not None]
        counts = [b["count"] for b in bins if b["count"] > 0]
        ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1, label="Perfect calibration")
        sizes = [20 + 300 * (c / max(counts)) for c in counts] if counts else []
        ax.scatter(xs, ys, s=sizes, color=colors[hz], zorder=3, label="MAZU model")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"{hz}\nECE={ece:.3f}  Brier={brier:.3f}")
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    chart_path = os.path.join(OUT_DIR, "calibration_reliability_diagram.png")
    plt.savefig(chart_path, dpi=140)
    plt.close()

    report_path = os.path.join(HERE, "calibration_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[SAVED] {chart_path}")
    print(f"[SAVED] {report_path}")
    for hz in ("heatwave", "flash_flood", "dust_storm"):
        r = report[hz]
        print(f"{hz}: ECE={r['ece']}  Brier={r['brier_score']}  n={r['n_test_samples']}")


if __name__ == "__main__":
    main()
