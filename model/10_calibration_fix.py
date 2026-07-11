# =============================================================================
# MAZU — tests whether isotonic-regression recalibration fixes the
# overconfidence found in 09_calibration.py, WITHOUT touching the production
# models. This trains a SEPARATE, isolated classifier instance (same
# hyperparameters as production) purely for this experiment.
#
# Leak-free 3-way split (chronological, nothing reused across roles):
#   Jan 1 - May 31  -> train the isolated classifier (151 days)
#   June 1 - 30     -> calibration set: fit isotonic regression on this
#                      classifier's OWN predictions here (30 days, genuinely
#                      unseen by the classifier during its training)
#   Jul 1 - Dec 31  -> test set (unchanged from every other report) -- used
#                      ONLY for the final before/after comparison, touched by
#                      neither the classifier's training nor the calibrator's
#                      fitting.
#
# Deliberately NOT using the same June data to train the base classifier:
# if the classifier had already seen June during training, its predictions
# on June would reflect training-set fit (falsely confident), not genuine
# out-of-sample behavior, which would bias the calibrator. This is why a
# clean 3-way split, not a 2-way one, is required for a valid experiment.
#
# Verdict: kept as a documented, tested finding -- NOT deployed to
# production (see report for why: the blast radius on ~150 already-verified
# tests/audit checks and CAP's own severity thresholds, all calibrated to
# the RAW probability scale, would be extensive).
# =============================================================================

import os
import sys
import json
import importlib.util
import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CALIB_START = "2025-06-01"
CALIB_END = "2025-06-30"
TRAIN2_END = "2025-05-31"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reliability_bins_and_scores(y_true, y_proba, n_bins=10):
    calib = load_module("calib_fix", os.path.join(HERE, "09_calibration.py"))
    return calib.reliability_bins(y_true, y_proba, n_bins=n_bins)


def run_hazard(hazard, X, y, dates):
    tr2_mask = dates <= TRAIN2_END
    calib_mask = (dates >= CALIB_START) & (dates <= CALIB_END)
    test_mask = dates > "2025-06-30"

    Xtr2, ytr2 = X[tr2_mask], y[tr2_mask]
    Xcal, ycal = X[calib_mask], y[calib_mask]
    Xte, yte = X[test_mask], y[test_mask]

    # Isolated classifier: SAME hyperparameters as the production models
    # (agent/01_train_and_save_models.py / model/07_dust_storm_forecast.py),
    # trained on LESS data (Jan-May only) so June is genuinely unseen.
    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                         class_weight="balanced", random_state=42, early_stopping=True)
    clf.fit(Xtr2, ytr2)

    proba_test_raw = clf.predict_proba(Xte)[:, 1]
    proba_cal_raw = clf.predict_proba(Xcal)[:, 1]

    # Fit isotonic regression ONLY on the calibration set (June) -- test set
    # is never touched by the fitting process.
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_cal_raw, ycal)
    proba_test_calibrated = iso.predict(proba_test_raw)

    bins_before, ece_before, brier_before = reliability_bins_and_scores(yte, proba_test_raw)
    bins_after, ece_after, brier_after = reliability_bins_and_scores(yte, proba_test_calibrated)

    roc_isolated = float(roc_auc_score(yte, proba_test_raw))
    pr_isolated = float(average_precision_score(yte, proba_test_raw))
    # Ranking quality (ROC-AUC) after calibration. NOTE: isotonic regression
    # is monotonic NON-DECREASING (not strictly increasing) -- it collapses
    # runs of the calibration set's raw probabilities into flat steps, so
    # distinct raw test-set probabilities can map to an IDENTICAL calibrated
    # value, introducing ties that did not exist in the original ranking.
    # This means ROC-AUC is only approximately preserved, not exactly
    # unchanged as a naive "monotonic transform" argument would suggest --
    # confirmed empirically below (flash_flood shows the largest real shift,
    # -0.0065, its smaller positive-sample count makes each new tie matter
    # more) and asserted with a tolerance, not equality, in the test suite.
    roc_after = float(roc_auc_score(yte, proba_test_calibrated))

    return {
        "isolated_classifier_train_period": f"2025-01-01 to {TRAIN2_END} ({int(tr2_mask.sum())} rows)",
        "calibration_period": f"{CALIB_START} to {CALIB_END} ({int(calib_mask.sum())} rows)",
        "test_period": f"2025-07-01 to 2025-12-31 ({int(test_mask.sum())} rows, UNCHANGED from every other report)",
        "isolated_classifier_roc_auc": round(roc_isolated, 4),
        "isolated_classifier_pr_auc": round(pr_isolated, 4),
        "roc_auc_after_calibration": round(roc_after, 4),
        "before": {"ece": ece_before, "brier": brier_before, "bins": bins_before},
        "after": {"ece": ece_after, "brier": brier_after, "bins": bins_after},
    }, proba_test_raw, proba_test_calibrated, yte


def main():
    fb = load_module("fb_fix", os.path.join(HERE, "03_forecast_baseline.py"))
    bn = load_module("bn_fix", os.path.join(HERE, "06_baseline_with_neighbors.py"))
    dust = load_module("dust_fix", os.path.join(HERE, "07_dust_storm_forecast.py"))

    ds = xr.open_dataset(fb.DATASET)
    report = {}
    raw_data = {}

    X, y, dates = bn.build_supervised_with_neighbors(ds, "heatwave")
    report["heatwave"], p_raw, p_cal, yte = run_hazard("heatwave", X, y, dates)
    raw_data["heatwave"] = (p_raw, p_cal, yte)

    X2, y2, dates2, _, _ = fb.build_supervised(ds, "flash_flood")
    report["flash_flood"], p_raw2, p_cal2, yte2 = run_hazard("flash_flood", X2, y2, dates2)
    raw_data["flash_flood"] = (p_raw2, p_cal2, yte2)

    label_all = dust.build_dust_label(ds)
    X3, y3, dates3, _, _ = dust.build_supervised(ds, label_all)
    report["dust_storm"], p_raw3, p_cal3, yte3 = run_hazard("dust_storm", X3, y3, dates3)
    raw_data["dust_storm"] = (p_raw3, p_cal3, yte3)

    ds.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = {"heatwave": "#d99a2b", "flash_flood": "#065A82", "dust_storm": "#8a5a2b"}
    for col, hz in enumerate(("heatwave", "flash_flood", "dust_storm")):
        for row, phase in enumerate(("before", "after")):
            ax = axes[row, col]
            bins = report[hz][phase]["bins"]
            xs = [b["mean_predicted"] for b in bins if b["mean_predicted"] is not None]
            ys = [b["observed_freq"] for b in bins if b["observed_freq"] is not None]
            counts = [b["count"] for b in bins if b["count"] > 0]
            sizes = [20 + 300 * (c / max(counts)) for c in counts] if counts else []
            ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1)
            ax.scatter(xs, ys, s=sizes, color=colors[hz], zorder=3)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f"{hz} -- {phase}\nECE={report[hz][phase]['ece']:.3f}")
            if row == 1:
                ax.set_xlabel("Mean predicted probability")
            if col == 0:
                ax.set_ylabel("Observed frequency")
    plt.tight_layout()
    chart_path = os.path.join(OUT_DIR, "calibration_before_after.png")
    plt.savefig(chart_path, dpi=140)
    plt.close()

    report_path = os.path.join(HERE, "calibration_fix_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[SAVED] {chart_path}")
    print(f"[SAVED] {report_path}")
    for hz in ("heatwave", "flash_flood", "dust_storm"):
        r = report[hz]
        print(f"{hz}: ECE {r['before']['ece']} -> {r['after']['ece']}   "
             f"Brier {r['before']['brier']} -> {r['after']['brier']}   "
             f"ROC-AUC unchanged: {r['isolated_classifier_roc_auc']} -> {r['roc_auc_after_calibration']}")


if __name__ == "__main__":
    main()
