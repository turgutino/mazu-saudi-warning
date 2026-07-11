# =============================================================================
# MAZU — tests whether an ensemble of independently-trained classifiers
# (bagging-style, different random seeds, same Jan-Jun training window as
# production) improves BOTH calibration (ECE/Brier) AND operational alert
# quality (POD/FAR/CSI/HSS at the SAME existing thresholds, 0.50/0.55 --
# unlike the isotonic-calibration experiment, no threshold re-derivation is
# attempted here, so this is a fair, direct, apples-to-apples comparison
# against the single production model).
#
# READ-ONLY on the real production models: this script loads
# agent/saved_models/*.joblib to get the single-model baseline, but never
# writes to that directory. All outputs go to NEW files
# (model/ensemble_report.json, outputs/ensemble_reliability_diagram.png) --
# nothing existing is modified.
# =============================================================================
import os
import sys
import json
import importlib.util
import joblib
import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(HERE, "..", "agent", "saved_models")
OUT_DIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_END = "2025-06-30"   # SAME production training window -- Jan 1 - Jun 30
SEEDS = [42, 43, 44, 45, 46]   # 5-member ensemble; seed 42 matches the single
                               # production model exactly (apples-to-apples)
THRESHOLDS = {"flash_flood": 0.50, "heatwave": 0.55, "dust_storm": 0.55}


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


def evaluate(proba, y_true, thr, calib_module):
    roc = float(roc_auc_score(y_true, proba))
    prauc = float(average_precision_score(y_true, proba))
    metrics = contingency_metrics(y_true, (proba >= thr).astype(int))
    _, ece, brier = calib_module.reliability_bins(y_true, proba)
    return {"roc_auc": round(roc, 4), "pr_auc": round(prauc, 4), **metrics,
           "ece": ece, "brier": brier}


def run_hazard(hazard, X, y, dates, saved_model_path, calib_module):
    tr_mask = dates <= TRAIN_END
    test_mask = dates > TRAIN_END
    Xtr, ytr = X[tr_mask], y[tr_mask]
    Xte, yte = X[test_mask], y[test_mask]

    # Single-model baseline: the ACTUAL saved production model (not a
    # retrained stand-in), so this comparison has zero "less training data"
    # caveat, unlike the isolated classifier used in the calibration
    # experiment.
    single_clf = joblib.load(saved_model_path)
    proba_single = single_clf.predict_proba(Xte)[:, 1]

    # Ensemble: N independently-trained classifiers, same hyperparameters
    # and training window as production, differing ONLY in random_state
    # (seed 42's member is trained identically to the saved production
    # model, so it is expected to reproduce it almost exactly -- checked
    # explicitly in the test suite as a sanity bypass).
    ensemble_probas = []
    for seed in SEEDS:
        clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                             class_weight="balanced", random_state=seed, early_stopping=True)
        clf.fit(Xtr, ytr)
        ensemble_probas.append(clf.predict_proba(Xte)[:, 1])
    ensemble_probas = np.array(ensemble_probas)
    proba_ensemble = ensemble_probas.mean(axis=0)

    thr = THRESHOLDS[hazard]
    single_eval = evaluate(proba_single, yte, thr, calib_module)
    ensemble_eval = evaluate(proba_ensemble, yte, thr, calib_module)

    print(f"{hazard}: single ROC={single_eval['roc_auc']} POD={single_eval['pod']} "
         f"CSI={single_eval['csi']} ECE={single_eval['ece']} Brier={single_eval['brier']}")
    print(f"{hazard}: ensemble(n={len(SEEDS)}) ROC={ensemble_eval['roc_auc']} POD={ensemble_eval['pod']} "
         f"CSI={ensemble_eval['csi']} ECE={ensemble_eval['ece']} Brier={ensemble_eval['brier']}")

    return {
        "threshold": thr, "n_ensemble_members": len(SEEDS), "seeds": SEEDS,
        "single_model": single_eval, "ensemble": ensemble_eval,
    }, proba_single, proba_ensemble, yte


def main():
    fb = load_module("fb_e", os.path.join(HERE, "03_forecast_baseline.py"))
    bn = load_module("bn_e", os.path.join(HERE, "06_baseline_with_neighbors.py"))
    dust = load_module("dust_e", os.path.join(HERE, "07_dust_storm_forecast.py"))
    calib_module = load_module("calib_e", os.path.join(HERE, "09_calibration.py"))

    ds = xr.open_dataset(fb.DATASET)
    report = {}
    plot_data = {}

    X, y, dates = bn.build_supervised_with_neighbors(ds, "heatwave")
    report["heatwave"], ps, pe, yte = run_hazard("heatwave", X, y, dates,
                                                   os.path.join(SAVED_DIR, "heatwave_model.joblib"), calib_module)
    plot_data["heatwave"] = (ps, pe, yte)

    X2, y2, dates2, _, _ = fb.build_supervised(ds, "flash_flood")
    report["flash_flood"], ps2, pe2, yte2 = run_hazard("flash_flood", X2, y2, dates2,
                                                          os.path.join(SAVED_DIR, "flash_flood_model.joblib"), calib_module)
    plot_data["flash_flood"] = (ps2, pe2, yte2)

    label_all = dust.build_dust_label(ds)
    X3, y3, dates3, _, _ = dust.build_supervised(ds, label_all)
    report["dust_storm"], ps3, pe3, yte3 = run_hazard("dust_storm", X3, y3, dates3,
                                                         os.path.join(SAVED_DIR, "dust_storm_model.joblib"), calib_module)
    plot_data["dust_storm"] = (ps3, pe3, yte3)

    ds.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = {"heatwave": "#d99a2b", "flash_flood": "#065A82", "dust_storm": "#8a5a2b"}
    for col, hz in enumerate(("heatwave", "flash_flood", "dust_storm")):
        ps, pe, yte = plot_data[hz]
        for row, (label, proba) in enumerate((("single", ps), ("ensemble", pe))):
            ax = axes[row, col]
            _, ece, _ = calib_module.reliability_bins(yte, proba)
            bins, _, _ = calib_module.reliability_bins(yte, proba)
            xs = [b["mean_predicted"] for b in bins if b["mean_predicted"] is not None]
            ys = [b["observed_freq"] for b in bins if b["observed_freq"] is not None]
            counts = [b["count"] for b in bins if b["count"] > 0]
            sizes = [20 + 300 * (c / max(counts)) for c in counts] if counts else []
            ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1)
            ax.scatter(xs, ys, s=sizes, color=colors[hz], zorder=3)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f"{hz} -- {label}\nECE={ece:.3f}")
            if row == 1:
                ax.set_xlabel("Mean predicted probability")
            if col == 0:
                ax.set_ylabel("Observed frequency")
    plt.tight_layout()
    chart_path = os.path.join(OUT_DIR, "ensemble_reliability_diagram.png")
    plt.savefig(chart_path, dpi=140)
    plt.close()

    report_path = os.path.join(HERE, "ensemble_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SAVED] {chart_path}")
    print(f"[SAVED] {report_path}")


if __name__ == "__main__":
    main()
