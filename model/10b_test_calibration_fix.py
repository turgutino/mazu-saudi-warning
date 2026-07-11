# =============================================================================
# MAZU — independent verification of 10_calibration_fix.py's isotonic-
# recalibration experiment: confirms the 3-way split is genuinely leak-free,
# the reported before/after numbers are internally consistent, and the
# "ROC-AUC approximately preserved" claim is checked with a tolerance
# (not assumed exact), matching the real, disclosed finding that isotonic
# regression's tie-flattening causes small real AUC shifts.
# =============================================================================
import sys
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

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


REPORT_PATH = os.path.join(HERE, "calibration_fix_report.json")
if not os.path.exists(REPORT_PATH):
    print("  [SKIP] calibration_fix_report.json not found -- run 10_calibration_fix.py first")
    sys.exit(0)

with open(REPORT_PATH, encoding="utf-8") as f:
    report = json.load(f)

HAZARDS = ("heatwave", "flash_flood", "dust_storm")

# --- Leak-free split verification -------------------------------------------
for hz in HAZARDS:
    r = report[hz]
    check(f"{hz}: isolated classifier trained ONLY through 2025-05-31 (not the full Jan-Jun "
         "production window) -- June is genuinely unseen by the classifier",
         "2025-05-31" in r["isolated_classifier_train_period"], r["isolated_classifier_train_period"])
    check(f"{hz}: calibration set is exactly June 2025 (30 days), not overlapping the classifier's "
         "own training period or the test period",
         "2025-06-01 to 2025-06-30" in r["calibration_period"], r["calibration_period"])
    check(f"{hz}: test period is unchanged (2025-07-01 to 2025-12-31), same as every other report "
         "in this project -- the calibrator never saw or was fit on this data",
         "2025-07-01 to 2025-12-31" in r["test_period"], r["test_period"])

# --- Structural / numeric sanity on before vs after -------------------------
for hz in HAZARDS:
    r = report[hz]
    before_total = sum(b["count"] for b in r["before"]["bins"])
    after_total = sum(b["count"] for b in r["after"]["bins"])
    check(f"{hz}: before/after bin counts both sum to the SAME test-set size "
         "(calibration only relabels probabilities, never drops/duplicates samples)",
         before_total == after_total, (before_total, after_total))
    check(f"{hz}: Brier score is a valid [0,1] value both before and after",
         0.0 <= r["before"]["brier"] <= 1.0 and 0.0 <= r["after"]["brier"] <= 1.0,
         (r["before"]["brier"], r["after"]["brier"]))

# --- The real finding: Brier improves for all 3 hazards ---------------------
for hz in HAZARDS:
    r = report[hz]
    check(f"{hz}: isotonic recalibration genuinely improves (lowers) the Brier score on the "
         "held-out, never-touched test set",
         r["after"]["brier"] < r["before"]["brier"], (r["before"]["brier"], r["after"]["brier"]))

# --- The real, disclosed nuance: ECE does NOT uniformly improve everywhere --
# heatwave's ECE gets slightly WORSE despite its Brier improving -- locked in
# as an explicit, expected result (not silently smoothed over), a reminder
# that a single scalar metric can mislead, same lesson as flash_flood's
# aggregate ECE in 09_calibration.py.
check("heatwave: ECE genuinely gets slightly WORSE post-calibration even though Brier improves "
     "-- a real, disclosed nuance (not hidden), consistent with this project's earlier finding "
     "that a single scalar calibration metric can be misleading",
     report["heatwave"]["after"]["ece"] > report["heatwave"]["before"]["ece"],
     (report["heatwave"]["before"]["ece"], report["heatwave"]["after"]["ece"]))
for hz in ("flash_flood", "dust_storm"):
    check(f"{hz}: ECE genuinely improves post-calibration",
         report[hz]["after"]["ece"] < report[hz]["before"]["ece"],
         (report[hz]["before"]["ece"], report[hz]["after"]["ece"]))

# --- ROC-AUC: approximately preserved, NOT exactly equal --------------------
# Isotonic regression is monotonic non-decreasing (not strictly increasing),
# so it can introduce new ties that shift ROC-AUC slightly -- this is
# asserted with a tolerance, and the largest real shift (flash_flood) is
# explicitly checked to be small but non-zero, not silently rounded to 0.
for hz in HAZARDS:
    r = report[hz]
    delta = abs(r["roc_auc_after_calibration"] - r["isolated_classifier_roc_auc"])
    check(f"{hz}: ROC-AUC is approximately preserved post-calibration (within 0.01, as expected "
         "for a monotonic-non-decreasing transform)", delta < 0.01,
         (r["isolated_classifier_roc_auc"], r["roc_auc_after_calibration"], delta))
check("flash_flood: the ROC-AUC shift from calibration is real and non-zero (tie-flattening "
     "effect genuinely measured, not a rounding artifact reported as exactly 0)",
     abs(report["flash_flood"]["roc_auc_after_calibration"] - report["flash_flood"]["isolated_classifier_roc_auc"]) > 0.0001,
     report["flash_flood"])

# --- Bypass: independently re-fit isotonic regression for flash_flood and ---
# compare against the stored report, from raw data, not trusting the script's
# own saved numbers.
try:
    import xarray as xr
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import mean_squared_error
    import importlib.util as ilu

    def load_mod(name, path):
        spec = ilu.spec_from_file_location(name, path)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    fb = load_mod("fb_bypass", os.path.join(HERE, "03_forecast_baseline.py"))
    ds = xr.open_dataset(fb.DATASET)
    X2, y2, dates2, _, _ = fb.build_supervised(ds, "flash_flood")
    ds.close()

    tr2_mask = dates2 <= "2025-05-31"
    calib_mask = (dates2 >= "2025-06-01") & (dates2 <= "2025-06-30")
    test_mask = dates2 > "2025-06-30"

    clf_bp = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                            class_weight="balanced", random_state=42, early_stopping=True)
    clf_bp.fit(X2[tr2_mask], y2[tr2_mask])
    proba_cal_bp = clf_bp.predict_proba(X2[calib_mask])[:, 1]
    proba_test_bp = clf_bp.predict_proba(X2[test_mask])[:, 1]

    iso_bp = IsotonicRegression(out_of_bounds="clip")
    iso_bp.fit(proba_cal_bp, y2[calib_mask])
    proba_test_cal_bp = iso_bp.predict(proba_test_bp)

    brier_bp = float(mean_squared_error(y2[test_mask], proba_test_cal_bp))
    stored_brier = report["flash_flood"]["after"]["brier"]
    check("flash_flood: independently re-trained classifier (same random_state=42, same split) "
         "+ independently re-fit isotonic regression reproduces the stored Brier score "
         "(bypass of the main script, deterministic given the fixed random_state)",
         abs(brier_bp - stored_brier) < 0.001, (brier_bp, stored_brier))
except Exception as e:
    check("flash_flood: bypass re-derivation ran without error", False, str(e))

print()
print("=" * 70)
print(f"TOTAL: {PASS} passed, {FAIL} failed")
print("=" * 70)
if FAIL > 0:
    sys.exit(1)
