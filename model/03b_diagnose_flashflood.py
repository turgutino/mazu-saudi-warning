# Diagnostic: is flash-flood forecast weak due to threshold choice, or genuinely
# low discriminative skill? Uses threshold-independent metrics (ROC-AUC, PR-AUC)
# and a full precision-recall sweep, re-using the exact same train/test split
# and features as 03_forecast_baseline.py.
import os
import importlib.util
import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fb", os.path.join(HERE, "03_forecast_baseline.py"))
fb = importlib.util.module_from_spec(spec)
import sys as _sys
_sys.modules["fb"] = fb
# don't execute main() on import — patch __name__ guard by loading module without running
src = open(os.path.join(HERE, "03_forecast_baseline.py"), encoding="utf-8").read()
src = src.replace('if __name__ == "__main__":\n    main()', '')
exec(compile(src, "fb", "exec"), fb.__dict__)

ds = xr.open_dataset(fb.DATASET)
X, y, dates, lat_flat, lon_flat = fb.build_supervised(ds, "flash_flood")
train_mask = dates <= fb.TRAIN_END
Xtr, ytr = X[train_mask], y[train_mask]
Xte, yte, dte = X[~train_mask], y[~train_mask], dates[~train_mask]

clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                     class_weight="balanced", random_state=42, early_stopping=True)
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)[:, 1]

roc = roc_auc_score(yte, proba)
pr_auc = average_precision_score(yte, proba)
base_rate = yte.mean()
print(f"flash_flood test base rate: {100*base_rate:.3f}%")
print(f"ROC-AUC: {roc:.3f}  (0.5=random, 1.0=perfect)")
print(f"PR-AUC:  {pr_auc:.3f}  (base rate {base_rate:.4f} = random baseline)")

prec, rec, thr = precision_recall_curve(yte, proba)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_i = np.argmax(f1[:-1])
print(f"\nBest-F1 threshold: {thr[best_i]:.4f}  Precision={prec[best_i]:.3f} Recall={rec[best_i]:.3f} F1={f1[best_i]:.3f}")

print("\nPrecision/Recall at fixed thresholds:")
for t in [0.5, 0.3, 0.2, 0.1, 0.05, thr[best_i]]:
    pred = (proba >= t).astype(int)
    tp = ((pred == 1) & (yte == 1)).sum(); fp = ((pred == 1) & (yte == 0)).sum()
    fn = ((pred == 0) & (yte == 1)).sum()
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    print(f"  thr={t:.3f}: P={p:.3f} R={r:.3f} F1={2*p*r/(p+r+1e-9):.3f}  (flags {pred.sum()} of {len(pred)} cells)")

# feature importance via permutation (cheap subsample)
from sklearn.inspection import permutation_importance
rng = np.random.default_rng(0)
sub = rng.choice(len(Xte), size=min(60000, len(Xte)), replace=False)
pi = permutation_importance(clf, Xte[sub], yte[sub], n_repeats=3, random_state=0, scoring="average_precision")
names = fb.FEATURE_VARS + ["lat", "lon", "day_of_year"]
order = np.argsort(pi.importances_mean)[::-1]
print("\nTop 8 features (permutation importance, PR-AUC drop):")
for i in order[:8]:
    print(f"  {names[i]:24s} {pi.importances_mean[i]:+.4f}")
ds.close()
