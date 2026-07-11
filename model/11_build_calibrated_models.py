# =============================================================================
# MAZU — build CALIBRATED production models: TESTED, NOT DEPLOYED.
#
# Kept here purely as a documented reference (same treatment as
# model/05_stgnn.py) -- running this script would overwrite the live
# agent/saved_models/*.joblib files, so it is NOT wired into any test,
# audit, or CI path, and was never run against this repo's actual
# saved_models/ (only against an isolated sandbox copy of the whole repo).
#
# Verdict, reached by actually building and testing this end-to-end in that
# sandbox (see agent/CALIBRATION_REPORT.md, "A full production-migration
# attempt..."): calibration genuinely fixed probability honesty (Brier score
# improved for all 3 hazards) but, even after properly re-deriving each
# hazard's operational alert threshold (CSI-maximized on a held-out
# calibration month, never the test set), real alert-issuance quality
# (POD/CSI/HSS) got WORSE for every hazard -- calibration and operational
# decision quality turned out to be separate concerns, and fixing one did
# not fix the other. Not deployed for that reason, not merely due to the
# blast-radius risk to ~150 existing verified numbers.
#
# For each hazard: trains an isolated classifier on Jan 1 - May 31 2025 (same
# hyperparameters as the original production training), fits an isotonic
# calibrator on June 2025 (genuinely unseen by the classifier), wraps both
# into a single CalibratedModel, and verifies ROC-AUC/PR-AUC/POD/FAR/CSI/HSS
# and calibration (ECE/Brier) on the SAME held-out Jul-Dec test set used
# everywhere else in this project -- before saving anything.
#
# This exactly reuses the leak-free 3-way split already validated in
# model/10_calibration_fix.py (24+26 independent tests passed there); this
# script's job is only to WRAP that already-tested approach into a
# deployable model object and refresh model_meta.json to match.
# =============================================================================
import os
import sys
import json
import importlib.util
import joblib
import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(HERE, "..", "agent")
SAVED_DIR = os.path.join(AGENT_DIR, "saved_models")

sys.path.insert(0, AGENT_DIR)
from calibrated_model import CalibratedModel

TRAIN2_END = "2025-05-31"
CALIB_START = "2025-06-01"
CALIB_END = "2025-06-30"

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def contingency_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    pod = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    far = fp / (tp + fp) if (tp + fp) > 0 else float("nan")
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else float("nan")
    n = tp + fp + fn + tn
    exp_correct = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / n if n > 0 else float("nan")
    hss_denom = n - exp_correct
    hss = (tp + tn - exp_correct) / hss_denom if hss_denom != 0 else float("nan")
    return {"pod": round(pod, 4), "far": round(far, 4), "csi": round(csi, 4), "hss": round(hss, 4)}


def select_threshold_by_csi(proba, y_true, candidates):
    """CSI-maximizing threshold selection -- standard meteorological
    forecast-verification practice (Wilks). Run ONLY on the calibration set
    (never the test set) so the final test-set POD/FAR/CSI/HSS report is not
    threshold-tuned on the same data it evaluates."""
    best_thr, best_csi = candidates[0], -1.0
    for thr in candidates:
        pred = (proba >= thr).astype(int)
        m = contingency_metrics(y_true, pred)
        if not np.isnan(m["csi"]) and m["csi"] > best_csi:
            best_csi, best_thr = m["csi"], thr
    return best_thr, best_csi


def build_hazard(hazard, X, y, dates, feature_list):
    tr2_mask = dates <= TRAIN2_END
    calib_mask = (dates >= CALIB_START) & (dates <= CALIB_END)
    test_mask = dates > CALIB_END

    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                         class_weight="balanced", random_state=42, early_stopping=True)
    clf.fit(X[tr2_mask], y[tr2_mask])

    proba_cal_raw = clf.predict_proba(X[calib_mask])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_cal_raw, y[calib_mask])

    wrapped = CalibratedModel(clf, iso)

    # Operational threshold is re-derived on the CALIBRATION set (June) only
    # -- reusing DetectionEngine's old rule-score thresholds (0.5/0.55) blindly
    # was found to be badly wrong for a calibrated probability scale (dust_storm's
    # calibrated probability never exceeds 0.5 on the test set at all, making the
    # old 0.55 threshold impossible to cross -- a real, investigated finding, not
    # a bug). CSI-maximizing threshold selection on held-out calibration data
    # (never the test set) is standard meteorological verification practice.
    proba_calib_for_thr = wrapped.predict_proba(X[calib_mask])[:, 1]
    candidates = np.round(np.arange(0.02, 0.51, 0.02), 2)
    new_thr, calib_csi = select_threshold_by_csi(proba_calib_for_thr, y[calib_mask], candidates)

    proba_test = wrapped.predict_proba(X[test_mask])[:, 1]
    yte = y[test_mask]
    roc = float(roc_auc_score(yte, proba_test))
    prauc = float(average_precision_score(yte, proba_test))
    thr = float(new_thr)
    metrics = contingency_metrics(yte, (proba_test >= thr).astype(int))
    print(f"  [{hazard}] threshold re-derived on calibration set (CSI={calib_csi:.3f} there): {thr}")

    calib_module = load_module("calib_b", os.path.join(HERE, "09_calibration.py"))
    bins, ece, brier = calib_module.reliability_bins(yte, proba_test)

    print(f"{hazard}: ROC-AUC={roc:.4f} PR-AUC={prauc:.4f} "
         f"POD={metrics['pod']} FAR={metrics['far']} CSI={metrics['csi']} HSS={metrics['hss']} "
         f"ECE={ece:.4f} Brier={brier:.4f}")

    return wrapped, {
        "features": feature_list,
        "roc_auc": round(roc, 4), "pr_auc": round(prauc, 4),
        "meteorological_metrics": {"threshold": thr, **metrics},
        "calibration": {"ece": round(ece, 4), "brier_score": round(brier, 4)},
        "calibrated": True,
        "isolated_classifier_train_period": f"2025-01-01 to {TRAIN2_END}",
        "calibration_period": f"{CALIB_START} to {CALIB_END}",
    }


def main():
    fb = load_module("fb_b", os.path.join(HERE, "03_forecast_baseline.py"))
    bn = load_module("bn_b", os.path.join(HERE, "06_baseline_with_neighbors.py"))
    dust = load_module("dust_b", os.path.join(HERE, "07_dust_storm_forecast.py"))

    ds = xr.open_dataset(fb.DATASET)
    meta = {}

    X, y, dates = bn.build_supervised_with_neighbors(ds, "heatwave")
    hw_features = fb.FEATURE_VARS + [f"neigh_{v}" for v in bn.NEIGHBOR_VARS] + ["lat", "lon", "day_of_year"]
    model_hw, meta["heatwave"] = build_hazard("heatwave", X, y, dates, hw_features)
    meta["heatwave"]["label_var"] = "heatwave_day_flag"
    meta["heatwave"]["label_thr"] = 1

    X2, y2, dates2, _, _ = fb.build_supervised(ds, "flash_flood")
    ff_features = fb.FEATURE_VARS + ["lat", "lon", "day_of_year"]
    model_ff, meta["flash_flood"] = build_hazard("flash_flood", X2, y2, dates2, ff_features)
    meta["flash_flood"]["label_var"] = "flash_flood_risk"
    meta["flash_flood"]["label_thr"] = 2

    label_all = dust.build_dust_label(ds)
    X3, y3, dates3, _, _ = dust.build_supervised(ds, label_all)
    dust_features = fb.FEATURE_VARS + ["wind10_speed", "dewpoint_depression_c", "lat", "lon", "day_of_year"]
    model_dust, meta["dust_storm"] = build_hazard("dust_storm", X3, y3, dates3, dust_features)
    meta["dust_storm"]["label_source"] = "DetectionEngine RULES['dust_storm'] (risk_threshold=0.55)"

    ds.close()

    meta["train_end"] = TRAIN2_END
    meta["calibration_period"] = f"{CALIB_START} to {CALIB_END}"
    meta["stride"] = 2
    meta["note"] = ("CALIBRATED models (v2): base classifiers trained Jan-May 2025 (1 month less "
                    "than the original v1 production models), isotonic-calibrated on June 2025, "
                    "tested on the SAME unchanged Jul-Dec 2025 test set as v1. See "
                    "agent/CALIBRATION_REPORT.md and agent/CALIBRATION_MIGRATION_REPORT.md.")

    joblib.dump(model_hw, os.path.join(SAVED_DIR, "heatwave_model.joblib"))
    joblib.dump(model_ff, os.path.join(SAVED_DIR, "flash_flood_model.joblib"))
    joblib.dump(model_dust, os.path.join(SAVED_DIR, "dust_storm_model.joblib"))
    with open(os.path.join(SAVED_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[SAVED] calibrated models + model_meta.json (SANDBOX)")


if __name__ == "__main__":
    main()
