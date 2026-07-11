# =============================================================================
# MAZU — train and SAVE 5 ensemble members per hazard (unlike
# model/12_ensemble.py, which trained members in-memory purely to compare
# single-vs-ensemble metrics and discarded them, this script persists all 15
# models to agent/saved_models/ensemble/ so forecast_tool can load them at
# inference time to compute a real ensemble-spread uncertainty estimate).
#
# Same training window as production (Jan 1 - Jun 30 2025, full 6 months --
# no held-out calibration month needed here, unlike the isotonic-calibration
# experiment) and same hyperparameters, differing ONLY in random_state.
# Seed 42 is intentionally included: it is trained identically to the actual
# production model, giving a built-in sanity check (its ROC-AUC must match
# the production model's own reported number).
# =============================================================================
import os
import sys
import json
import time
import importlib.util
import joblib
import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(HERE, "..", "agent", "saved_models")
ENS_DIR = os.path.join(SAVED_DIR, "ensemble")
os.makedirs(ENS_DIR, exist_ok=True)

TRAIN_END = "2025-06-30"
SEEDS = [42, 43, 44, 45, 46]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def train_hazard(hazard, X, y, dates):
    tr_mask = dates <= TRAIN_END
    test_mask = dates > TRAIN_END
    Xtr, ytr = X[tr_mask], y[tr_mask]
    Xte, yte = X[test_mask], y[test_mask]

    results = []
    for seed in SEEDS:
        t0 = time.time()
        clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                             class_weight="balanced", random_state=seed, early_stopping=True)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        roc = float(roc_auc_score(yte, proba))
        prauc = float(average_precision_score(yte, proba))
        elapsed = time.time() - t0

        fname = f"{hazard}_seed{seed}.joblib"
        joblib.dump(clf, os.path.join(ENS_DIR, fname))
        print(f"  [{hazard}] seed={seed} ROC-AUC={roc:.4f} PR-AUC={prauc:.4f} ({elapsed:.0f}s) -> {fname}")
        results.append({"seed": seed, "roc_auc": round(roc, 4), "pr_auc": round(prauc, 4), "file": fname})

    return results


def main():
    fb = load_module("fb_ens", os.path.join(HERE, "03_forecast_baseline.py"))
    bn = load_module("bn_ens", os.path.join(HERE, "06_baseline_with_neighbors.py"))
    dust = load_module("dust_ens", os.path.join(HERE, "07_dust_storm_forecast.py"))

    ds = xr.open_dataset(fb.DATASET)
    manifest = {"seeds": SEEDS, "train_end": TRAIN_END, "hazards": {}}

    print("### heatwave ###")
    X, y, dates = bn.build_supervised_with_neighbors(ds, "heatwave")
    manifest["hazards"]["heatwave"] = train_hazard("heatwave", X, y, dates)

    print("### flash_flood ###")
    X2, y2, dates2, _, _ = fb.build_supervised(ds, "flash_flood")
    manifest["hazards"]["flash_flood"] = train_hazard("flash_flood", X2, y2, dates2)

    print("### dust_storm ###")
    label_all = dust.build_dust_label(ds)
    X3, y3, dates3, _, _ = dust.build_supervised(ds, label_all)
    manifest["hazards"]["dust_storm"] = train_hazard("dust_storm", X3, y3, dates3)

    ds.close()

    manifest_path = os.path.join(ENS_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[SAVED] {manifest_path}")
    print("[DONE] all 15 ensemble members saved to", ENS_DIR)


if __name__ == "__main__":
    main()
