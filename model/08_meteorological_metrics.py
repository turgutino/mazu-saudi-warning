# =============================================================================
# MAZU — meteorological verification metrics (POD, FAR, CSI, HSS)
#
# ROC-AUC/PR-AUC (already reported) are threshold-independent and correct for
# rare-event classification. This script adds the complementary, WMO/national-
# met-service-standard contingency-table metrics computed at each hazard's own
# OPERATIONAL threshold -- the exact same threshold already used elsewhere in
# this codebase for CAP alert severity (DetectionEngine.RULES[hazard]
# ["severity"][1][1]: 0.50 for flash_flood, 0.55 for heatwave/dust_storm),
# not a newly invented cutoff.
#
# Uses the ALREADY-SAVED, already-verified production models (joblib) --
# does NOT retrain -- so there is zero risk of these numbers drifting from
# what the agent actually serves. Rebuilds the exact same held-out Jul-Dec
# test set each model was originally verified against (same code paths as
# agent/01_train_and_save_models.py and model/07_dust_storm_forecast.py).
# =============================================================================

import os
import sys
import json
import importlib.util
import joblib
import numpy as np
import xarray as xr
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(HERE, "..", "agent", "saved_models")


def load_module(name, path):
    # No source truncation needed: module_from_spec sets mod.__dict__["__name__"]
    # to `name` (not "__main__"), so each file's own `if __name__ == "__main__":`
    # guard naturally does not fire -- unlike a naive string-replace/split
    # approach, this is also safe for files (06_baseline_with_neighbors.py)
    # that themselves contain that exact guard string inside a nested
    # string literal.
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def contingency_metrics(y_true, y_pred):
    """Standard 2x2 contingency-table metrics (WMO verification standard).
    tn, fp, fn, tp order matches sklearn.metrics.confusion_matrix.ravel()."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    pod = tp / (tp + fn) if (tp + fn) > 0 else float("nan")          # hit rate / recall
    far = fp / (tp + fp) if (tp + fp) > 0 else float("nan")          # false alarm ratio
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else float("nan")  # critical success index
    n = tp + fp + fn + tn
    exp_correct = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / n if n > 0 else float("nan")
    hss_denom = n - exp_correct
    hss = (tp + tn - exp_correct) / hss_denom if hss_denom != 0 else float("nan")
    return {"pod": pod, "far": far, "csi": csi, "hss": hss,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def main():
    de = load_module("de", os.path.join(HERE, "01_detection_engine.py"))
    fb = load_module("fb", os.path.join(HERE, "03_forecast_baseline.py"))
    bn = load_module("bn", os.path.join(HERE, "06_baseline_with_neighbors.py"))
    dust = load_module("dust", os.path.join(HERE, "07_dust_storm_forecast.py"))

    ds = xr.open_dataset(fb.DATASET)

    # Operational threshold per hazard = this hazard's own 2nd severity tier
    # lower bound (the same numbers already used for CAP alert severity in
    # agent/tools.py's _cap_severity -- not a second, independently invented
    # cutoff scale).
    thresholds = {hz: de.RULES[hz]["severity"][1][1] for hz in ("flash_flood", "heatwave", "dust_storm")}

    results = {}
    lines = ["=" * 70, "MAZU -- meteorological verification metrics (POD/FAR/CSI/HSS)",
             "=" * 70,
             "Computed from the ALREADY-SAVED production models (no retraining) on",
             "the same held-out Jul-Dec test set each was originally verified on.",
             ""]

    # ── heatwave (neighbour-feature model) ──────────────────────────────────
    X, y, dates = bn.build_supervised_with_neighbors(ds, "heatwave")
    te = dates > fb.TRAIN_END
    clf = joblib.load(os.path.join(SAVED_DIR, "heatwave_model.joblib"))
    proba = clf.predict_proba(X[te])[:, 1]
    thr = thresholds["heatwave"]
    m = contingency_metrics(y[te], (proba >= thr).astype(int))
    results["heatwave"] = {"threshold": thr, **m}

    # ── flash_flood (plain baseline model) ──────────────────────────────────
    X2, y2, dates2, _lat2, _lon2 = fb.build_supervised(ds, "flash_flood")
    te2 = dates2 > fb.TRAIN_END
    clf2 = joblib.load(os.path.join(SAVED_DIR, "flash_flood_model.joblib"))
    proba2 = clf2.predict_proba(X2[te2])[:, 1]
    thr2 = thresholds["flash_flood"]
    m2 = contingency_metrics(y2[te2], (proba2 >= thr2).astype(int))
    results["flash_flood"] = {"threshold": thr2, **m2}

    # ── dust_storm ────────────────────────────────────────────────────────
    label_all = dust.build_dust_label(ds)
    X3, y3, dates3, _lat3, _lon3 = dust.build_supervised(ds, label_all)
    te3 = dates3 > dust.TRAIN_END
    clf3 = joblib.load(os.path.join(SAVED_DIR, "dust_storm_model.joblib"))
    proba3 = clf3.predict_proba(X3[te3])[:, 1]
    thr3 = thresholds["dust_storm"]
    m3 = contingency_metrics(y3[te3], (proba3 >= thr3).astype(int))
    results["dust_storm"] = {"threshold": thr3, **m3}

    ds.close()

    for hz in ("flash_flood", "heatwave", "dust_storm"):
        r = results[hz]
        lines.append(f"--- {hz} (operational threshold={r['threshold']:.2f}) ---")
        lines.append(f"Confusion: TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']}")
        lines.append(f"POD={r['pod']:.3f}  FAR={r['far']:.3f}  CSI={r['csi']:.3f}  HSS={r['hss']:.3f}")
        lines.append("")

    rpt = "\n".join(lines)
    with open(os.path.join(HERE, "meteorological_metrics_report.txt"), "w", encoding="utf-8") as f:
        f.write(rpt)
    print(rpt)

    # ── merge into model_meta.json (adds fields, does not touch roc_auc/pr_auc) ──
    meta_path = os.path.join(SAVED_DIR, "model_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    for hz, r in results.items():
        meta[hz]["meteorological_metrics"] = {
            "threshold": round(r["threshold"], 2),
            "pod": round(r["pod"], 4), "far": round(r["far"], 4),
            "csi": round(r["csi"], 4), "hss": round(r["hss"], 4),
        }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[SAVED] {meta_path} (added meteorological_metrics per hazard)")


if __name__ == "__main__":
    main()
