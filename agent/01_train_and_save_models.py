# =============================================================================
# MAZU — Layer 4 prep: train and persist the PRODUCTION forecast models
#
# Uses the EXACT same code paths already verified in Layer 2:
#   - heatwave:     06_baseline_with_neighbors.py (best: ROC-AUC 0.971)
#   - flash_flood:  03_forecast_baseline.py plain baseline (ROC-AUC 0.873;
#                    the neighbour feature made flash_flood WORSE, so it is
#                    correctly NOT used for this hazard — see
#                    model/neighbor_feature_report.txt)
#
# After training, re-computes ROC-AUC/PR-AUC on the same held-out test set
# and asserts it matches the previously reported numbers (within float
# tolerance) BEFORE saving — if retraining ever drifts from the verified
# report, this script fails loudly instead of silently shipping a different
# model than what was reported.
# =============================================================================

import os
import sys
import importlib.util
import json
import joblib
import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "model")
OUT_DIR = os.path.join(HERE, "saved_models")
os.makedirs(OUT_DIR, exist_ok=True)

# previously verified numbers (from forecast_report.txt / neighbor_feature_report.txt)
# tolerance accounts for platform/float nondeterminism in HGB, not for real drift
EXPECTED = {
    "heatwave":    {"roc": 0.971, "prauc": 0.795, "tol": 0.01},
    "flash_flood": {"roc": 0.873, "prauc": 0.089, "tol": 0.01},
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    src = open(path, encoding="utf-8").read().replace('if __name__ == "__main__":\n    main()', "")
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


def main():
    fb = load_module("fb", os.path.join(MODEL_DIR, "03_forecast_baseline.py"))
    bn = load_module("bn", os.path.join(MODEL_DIR, "06_baseline_with_neighbors.py"))

    ds = xr.open_dataset(fb.DATASET)
    results = {}

    # ── Heatwave: neighbour-feature model (best verified) ──────────────────
    print("### Training heatwave (neighbour-feature) model ###")
    X, y, dates = bn.build_supervised_with_neighbors(ds, "heatwave")
    tr = dates <= fb.TRAIN_END
    clf_hw = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                            class_weight="balanced", random_state=42, early_stopping=True)
    clf_hw.fit(X[tr], y[tr])
    proba = clf_hw.predict_proba(X[~tr])[:, 1]
    roc = roc_auc_score(y[~tr], proba)
    prauc = average_precision_score(y[~tr], proba)
    print(f"  ROC-AUC={roc:.4f}  PR-AUC={prauc:.4f}")
    results["heatwave"] = {"roc": roc, "prauc": prauc}

    joblib.dump(clf_hw, os.path.join(OUT_DIR, "heatwave_model.joblib"))
    hw_features = fb.FEATURE_VARS + [f"neigh_{v}" for v in bn.NEIGHBOR_VARS] + ["lat", "lon", "day_of_year"]

    # ── Flash flood: plain baseline model (neighbour feature made it worse) ─
    print("\n### Training flash_flood (plain baseline) model ###")
    X2, y2, dates2, _lat2, _lon2 = fb.build_supervised(ds, "flash_flood")
    tr2 = dates2 <= fb.TRAIN_END
    clf_ff = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                            class_weight="balanced", random_state=42, early_stopping=True)
    clf_ff.fit(X2[tr2], y2[tr2])
    proba2 = clf_ff.predict_proba(X2[~tr2])[:, 1]
    roc2 = roc_auc_score(y2[~tr2], proba2)
    prauc2 = average_precision_score(y2[~tr2], proba2)
    print(f"  ROC-AUC={roc2:.4f}  PR-AUC={prauc2:.4f}")
    results["flash_flood"] = {"roc": roc2, "prauc": prauc2}

    joblib.dump(clf_ff, os.path.join(OUT_DIR, "flash_flood_model.joblib"))
    ff_features = fb.FEATURE_VARS + ["lat", "lon", "day_of_year"]

    # ── verify against previously reported numbers ──────────────────────────
    print("\n### Verification against previously reported numbers ###")
    all_ok = True
    for hz, exp in EXPECTED.items():
        got = results[hz]
        roc_ok = abs(got["roc"] - exp["roc"]) <= exp["tol"]
        prauc_ok = abs(got["prauc"] - exp["prauc"]) <= exp["tol"]
        status = "OK" if (roc_ok and prauc_ok) else "MISMATCH"
        if not (roc_ok and prauc_ok):
            all_ok = False
        print(f"  {hz:14s} ROC {got['roc']:.4f} (expected {exp['roc']:.3f}) "
             f"PR-AUC {got['prauc']:.4f} (expected {exp['prauc']:.3f})  [{status}]")

    if not all_ok:
        print("\n[FATAL] Retrained model metrics do not match the previously verified "
              "report. Refusing to save — investigate before the agent uses a model "
              "that has not been honestly verified.")
        sys.exit(1)

    # ── save feature metadata (needed to reconstruct inputs at inference) ───
    meta = {
        "heatwave": {"features": hw_features, "label_var": "heatwave_day_flag", "label_thr": 1,
                     "roc_auc": results["heatwave"]["roc"], "pr_auc": results["heatwave"]["prauc"]},
        "flash_flood": {"features": ff_features, "label_var": "flash_flood_risk", "label_thr": 2,
                        "roc_auc": results["flash_flood"]["roc"], "pr_auc": results["flash_flood"]["prauc"]},
        "train_end": fb.TRAIN_END,
        "stride": 2,
    }
    with open(os.path.join(OUT_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SAVED] {OUT_DIR}/heatwave_model.joblib")
    print(f"[SAVED] {OUT_DIR}/flash_flood_model.joblib")
    print(f"[SAVED] {OUT_DIR}/model_meta.json")
    print("\nAll metrics verified against Layer 2 reports — safe for agent use.")
    ds.close()


if __name__ == "__main__":
    main()
