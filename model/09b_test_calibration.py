# =============================================================================
# MAZU — independent verification of 09_calibration.py's reliability-diagram
# math BEFORE trusting calibration_report.json, matching this project's
# established pattern: verify every piece in isolation, with hand-computed
# and bypass checks, not just re-reading the script's own output.
# =============================================================================
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("calib", os.path.join(os.path.dirname(__file__), "09_calibration.py"))
calib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calib)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# --- Synthetic, hand-verifiable case 1: perfectly calibrated model -----------
# 1000 samples: probability p, true outcome is Bernoulli(p) on average, exact
# construction so every bin's observed_freq should equal its mean_predicted.
rng = np.random.default_rng(42)
y_proba_perfect = np.repeat(np.linspace(0.05, 0.95, 10), 1000)
y_true_perfect = (rng.random(len(y_proba_perfect)) < y_proba_perfect).astype(int)
bins, ece, brier = calib.reliability_bins(y_true_perfect, y_proba_perfect, n_bins=10)
check("perfectly-calibrated-by-construction case: ECE is small (<0.03, allowing for finite-sample noise)",
     ece < 0.03, ece)
check("perfectly-calibrated case: all 10 bins have count 1000 (exact bin construction)",
     all(b["count"] == 1000 for b in bins), [b["count"] for b in bins])

# --- Synthetic case 2: known, hand-computed overconfident model -------------
# Model always predicts 0.9, but true event only happens 30% of the time.
# Single bin [0.9,1.0] should show mean_predicted=0.9, observed_freq=0.3 exactly.
y_proba_over = np.full(500, 0.9)
y_true_over = np.array([1] * 150 + [0] * 350)  # exactly 30% positive
bins2, ece2, brier2 = calib.reliability_bins(y_true_over, y_proba_over, n_bins=10)
bin_90 = next(b for b in bins2 if b["bin_lo"] == 0.9)
check("hand-constructed overconfident case: bin[0.9,1.0] mean_predicted is exactly 0.9",
     abs(bin_90["mean_predicted"] - 0.9) < 1e-9, bin_90)
check("hand-constructed overconfident case: bin[0.9,1.0] observed_freq is exactly 0.3",
     abs(bin_90["observed_freq"] - 0.3) < 1e-9, bin_90)
check("hand-constructed overconfident case: ECE equals |0.9-0.3|=0.6 exactly (single populated bin)",
     abs(ece2 - 0.6) < 1e-9, ece2)
check("hand-constructed overconfident case: Brier score matches hand formula mean((p-y)^2)",
     abs(brier2 - np.mean((y_proba_over - y_true_over) ** 2)) < 1e-9, brier2)

# --- Synthetic case 3: empty bins are reported, not silently dropped -------
y_proba_sparse = np.array([0.02] * 100 + [0.97] * 5)
y_true_sparse = np.array([0] * 100 + [1, 1, 0, 1, 0])
bins3, ece3, brier3 = calib.reliability_bins(y_true_sparse, y_proba_sparse, n_bins=10)
empty_bins = [b for b in bins3 if b["count"] == 0]
check("sparse case: middle bins (no samples) are present with count=0, not omitted from the list",
     len(empty_bins) == 8, len(empty_bins))
check("sparse case: empty bins have mean_predicted=None and observed_freq=None (not fabricated as 0)",
     all(b["mean_predicted"] is None and b["observed_freq"] is None for b in empty_bins),
     empty_bins)
check("sparse case: total count across all bins equals sample size (no samples dropped)",
     sum(b["count"] for b in bins3) == len(y_true_sparse), sum(b["count"] for b in bins3))

# --- Real report: structural checks against calibration_report.json --------
REPORT_PATH = os.path.join(os.path.dirname(__file__), "calibration_report.json")
if os.path.exists(REPORT_PATH):
    with open(REPORT_PATH, encoding="utf-8") as f:
        report = json.load(f)
    for hz in ("heatwave", "flash_flood", "dust_storm"):
        check(f"{hz}: present in calibration_report.json", hz in report, list(report.keys()))
        if hz in report:
            r = report[hz]
            total_binned = sum(b["count"] for b in r["bins"])
            check(f"{hz}: sum of all bin counts equals n_test_samples (no sample dropped/duplicated)",
                 total_binned == r["n_test_samples"], (total_binned, r["n_test_samples"]))
            check(f"{hz}: ECE is a valid [0,1] value", 0.0 <= r["ece"] <= 1.0, r["ece"])
            check(f"{hz}: Brier score is a valid [0,1] value (binary outcome)", 0.0 <= r["brier_score"] <= 1.0, r["brier_score"])
    # Real, disclosed finding: all 3 hazards are OVERCONFIDENT at high
    # probability bins (points fall below the diagonal) -- lock this in as
    # an explicit assertion, not just an eyeballed chart observation.
    for hz in ("heatwave", "flash_flood", "dust_storm"):
        top_bin = report[hz]["bins"][-1]  # [0.9, 1.0]
        check(f"{hz}: top bin [0.9,1.0] shows genuine overconfidence "
             "(observed_freq well below mean_predicted -- a real, disclosed calibration gap)",
             top_bin["observed_freq"] is not None and top_bin["observed_freq"] < top_bin["mean_predicted"] - 0.1,
             top_bin)
else:
    print("  [SKIP] calibration_report.json not found -- run 09_calibration.py first")

print()
print("=" * 70)
print(f"TOTAL: {PASS} passed, {FAIL} failed")
print("=" * 70)
if FAIL > 0:
    sys.exit(1)
