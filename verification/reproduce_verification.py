# =============================================================================
# MAZU-FENGYUN — Independent reproduction of every real/model number cited in
# MAZU_Real_Hadise_Dogrulama_EN.html (12-event verification report).
#
# This script does NOT read the report. It recomputes every risk-score /
# model-probability value directly from real_grid()/predicted_grid()
# (build_grid.py), which themselves read straight from DetectionEngine and
# forecast_tool's model artifacts. The claimed values below were transcribed
# by hand from the published report; the script fails loudly on any mismatch.
#
# Scope: only risk-score / model-probability claims are checked here (i.e.
# anything real_grid/predicted_grid can produce). Raw meteorological values
# quoted in the report (e.g. "7.7 mm precipitation", "44.2C max temp", wind
# speed) are NOT risk scores and are out of scope for this script — they are
# listed in NOT_CHECKED below for transparency, not silently skipped.
#
# Usage:  python reproduce_verification.py
# Exit code: 0 if every claim matches within tolerance, 1 otherwise.
# =============================================================================

import sys
import numpy as np
from build_grid import real_grid, predicted_grid
import tools

TOL = 0.02  # absolute tolerance vs. the report's rounded figures

# Non-MAZU-city coordinates researched via WebSearch earlier in the project.
HAIL = (27.52, 41.70)
BURAIDAH = (26.33, 43.97)

C = tools.CITIES  # {"Dammam": (lat, lon), "Riyadh": ..., ...}

# (event_label, date, hazard, (lat, lon), claimed_real, claimed_model)
# claimed_real / claimed_model = None means "not asserted as a risk score in
# the report for this exact date/location" — skip that side of the check.
CLAIMS = [
    # --- Event 1: Dammam dust storm, 17-19 May (mid-term report + verification report) ---
    ("1. Dammam dust 17 May",  "2025-05-17", "dust_storm", C["Dammam"], 0.60, 0.94),
    ("1. Dammam dust 18 May",  "2025-05-18", "dust_storm", C["Dammam"], None, 0.91),
    ("1. Dammam dust 19 May",  "2025-05-19", "dust_storm", C["Dammam"], None, 0.05),

    # --- Event 2: Dammam heatwave, 13 June (corrected finding: real=0.0 all week) ---
    ("2. Dammam heat 10 Jun",  "2025-06-10", "heatwave",   C["Dammam"], 0.0,  None),
    ("2. Dammam heat 11 Jun",  "2025-06-11", "heatwave",   C["Dammam"], 0.0,  None),
    ("2. Dammam heat 12 Jun",  "2025-06-12", "heatwave",   C["Dammam"], 0.0,  None),
    ("2. Dammam heat 13 Jun",  "2025-06-13", "heatwave",   C["Dammam"], 0.0,  0.67),
    ("2. Dammam heat 14 Jun",  "2025-06-14", "heatwave",   C["Dammam"], 0.0,  None),
    ("2. Dammam heat 15 Jun",  "2025-06-15", "heatwave",   C["Dammam"], 0.0,  None),
    ("2. Dammam heat 16 Jun",  "2025-06-16", "heatwave",   C["Dammam"], 0.0,  None),

    # --- Event 3: Jeddah historic flood, 9-10 Dec (MISS — data resolution limit) ---
    ("3. Jeddah flood 9 Dec",  "2025-12-09", "flash_flood", C["Jeddah"], None, None),  # model max 0.15 across window; not a single-day figure, skipped

    # --- Event 4: Haboob dust, 4-5 May (Inconclusive, outside coverage) ---
    ("4. Jeddah haboob 4 May", "2025-05-04", "dust_storm", C["Jeddah"], 0.35, None),

    # --- Event 5: Dust storm, 30 Jun-5 Jul (Strong HIT) ---
    ("5. Riyadh dust 30 Jun",  "2025-06-30", "dust_storm", C["Riyadh"], 1.00, 0.985),
    ("5. Dammam dust 3 Jul",   "2025-07-03", "dust_storm", C["Dammam"], None, None),  # range only in report; spot-checked separately

    # --- Event 6: Heatwave 2nd wave, 28 Jun-5 Jul (MISS) ---
    ("6. Dammam heat2 30 Jun", "2025-06-30", "heatwave",   C["Dammam"], 0.55, 0.045),

    # --- Event 7: Heatwave 3rd wave, 20-27 Jul (Partial hit at Mecca) ---
    ("7. Mecca heat3 22 Jul",  "2025-07-22", "heatwave",   C["Mecca"],  0.60, 0.97),
    ("7. Mecca heat3 23 Jul",  "2025-07-23", "heatwave",   C["Mecca"],  0.45, 0.98),

    # --- Event 8: Heatwave 4th wave, 29 Jul-5 Aug (Partial hit) ---
    ("8. Mecca heat4 4 Aug",   "2025-08-04", "heatwave",   C["Mecca"],  0.60, 0.90),

    # --- Event 9: Flood, 6-7 Jan (Inconclusive — data gap + training period) ---
    ("9. Mecca flood 6 Jan",   "2025-01-06", "flash_flood", C["Mecca"], 1.00, 0.21),

    # --- Event 10: Flood, 6-7 Mar (Hail/Buraidah — MISS + new finding) ---
    ("10. Hail flood 6 Mar",     "2025-03-06", "flash_flood", HAIL,      0.0, 0.973),
    ("10. Hail flood 7 Mar",     "2025-03-07", "flash_flood", HAIL,      0.0, 0.978),
    ("10. Buraidah flood 6 Mar", "2025-03-06", "flash_flood", BURAIDAH,  0.0, 0.942),
    ("10. Buraidah flood 7 Mar", "2025-03-07", "flash_flood", BURAIDAH,  0.0, 0.816),

    ("8. Dammam heat4 5 Aug",  "2025-08-05", "heatwave", C["Dammam"], 0.2, None),

    # --- Event 11: Flood, 14 Aug (Taif — model/real disagreement) ---
    ("11. Taif flood 13 Aug",  "2025-08-13", "flash_flood", C["Taif"],  0.35, 0.52),
    ("11. Taif flood 14 Aug",  "2025-08-14", "flash_flood", C["Taif"],  0.15, 0.51),

    # --- Event 12: Flood, 27-28 Aug (Asir/Jizan — mixed result) ---
    ("12. Abha flood 26 Aug",  "2025-08-26", "flash_flood", C["Abha"],  0.75, 0.84),
    ("12. Jizan flood 26 Aug", "2025-08-26", "flash_flood", C["Jizan"], 0.60, None),
]

# (event_label, dates, hazard, (lat, lon), claimed_real_range, claimed_model_range)
# Range claims from the report, checked as: every daily value must fall within
# [lo-TOL, hi+TOL]. None means that side wasn't asserted as a range.
RANGE_CLAIMS = [
    ("5. Dammam dust 1-5 Jul",   ["2025-07-01","2025-07-02","2025-07-03","2025-07-04","2025-07-05"],
     "dust_storm", C["Dammam"], (0.35, 0.60), (0.77, 0.91)),
    ("8. Dammam heat4 2-5 Aug",  ["2025-08-02","2025-08-03","2025-08-04","2025-08-05"],
     "heatwave", C["Dammam"], None, (0.01, 0.05)),
    ("12. Abha flood 28-29 Aug", ["2025-08-28","2025-08-29"],
     "flash_flood", C["Abha"], (0.75, 0.75), (0.37, 0.44)),
    ("12. Jizan flood 26-27 Aug",["2025-08-26","2025-08-27"],
     "flash_flood", C["Jizan"], (0.60, 0.60), (0.18, 0.22)),
]

NOT_CHECKED = [
    "Event 1, 19 May: Dammam raw wind speed 2.1 m/s (not a risk score)",
    "Event 2, 13/16 Jun: Dammam raw tmax 40.0C / 44.2C (not a risk score)",
    "Event 3: Jeddah raw precipitation 7.7 mm / 31.6 mm regional max (not a risk score)",
    "Event 4: wider-grid high-risk zone 0.6-0.8 near Sudan/Jordan/Iraq border (region, not one point)",
]


def grid_at(lat_arr, lon_arr, grid, lat, lon):
    yi = int(np.argmin(np.abs(lat_arr - lat)))
    xi = int(np.argmin(np.abs(lon_arr - lon)))
    return float(grid[yi, xi])


def main():
    failures = []
    checked = 0
    for label, date, hazard, (lat, lon), claimed_real, claimed_model in CLAIMS:
        rlat, rlon, rgrid = real_grid(date, hazard)
        plat, plon, pgrid = predicted_grid(date, hazard)
        real_v = grid_at(rlat, rlon, rgrid, lat, lon)
        pred_v = grid_at(plat, plon, pgrid, lat, lon)

        line = f"{label:28s} {date} {hazard:12s} real={real_v:.3f}"
        if claimed_real is not None:
            checked += 1
            ok = abs(real_v - claimed_real) <= TOL
            line += f" (claim {claimed_real}) {'OK' if ok else '<<< MISMATCH'}"
            if not ok:
                failures.append(f"{label}: real {real_v:.3f} vs claimed {claimed_real}")

        line += f"   model={pred_v:.3f}"
        if claimed_model is not None:
            checked += 1
            ok = abs(pred_v - claimed_model) <= TOL
            line += f" (claim {claimed_model}) {'OK' if ok else '<<< MISMATCH'}"
            if not ok:
                failures.append(f"{label}: model {pred_v:.3f} vs claimed {claimed_model}")

        print(line)

    print("\n--- range claims ---")
    for label, dates, hazard, (lat, lon), real_range, model_range in RANGE_CLAIMS:
        real_vals, pred_vals = [], []
        for date in dates:
            rlat, rlon, rgrid = real_grid(date, hazard)
            plat, plon, pgrid = predicted_grid(date, hazard)
            real_vals.append(grid_at(rlat, rlon, rgrid, lat, lon))
            pred_vals.append(grid_at(plat, plon, pgrid, lat, lon))

        line = f"{label:28s} {dates[0]}..{dates[-1]} {hazard:12s}"
        if real_range is not None:
            checked += 1
            lo, hi = real_range
            ok = all(lo - TOL <= v <= hi + TOL for v in real_vals)
            line += f" real=[{min(real_vals):.3f},{max(real_vals):.3f}] (claim {real_range}) {'OK' if ok else '<<< MISMATCH'}"
            if not ok:
                failures.append(f"{label}: real values {[round(v,3) for v in real_vals]} outside claimed range {real_range}")
        if model_range is not None:
            checked += 1
            lo, hi = model_range
            ok = all(lo - TOL <= v <= hi + TOL for v in pred_vals)
            line += f"  model=[{min(pred_vals):.3f},{max(pred_vals):.3f}] (claim {model_range}) {'OK' if ok else '<<< MISMATCH'}"
            if not ok:
                failures.append(f"{label}: model values {[round(v,3) for v in pred_vals]} outside claimed range {model_range}")
        print(line)

    print()
    print(f"Checked {checked} individual claims across {len(CLAIMS)} date/location rows + {len(RANGE_CLAIMS)} range rows.")
    print(f"NOT independently checked by this script ({len(NOT_CHECKED)} items, raw-variable or range claims):")
    for n in NOT_CHECKED:
        print(f"  - {n}")

    if failures:
        print(f"\n{len(failures)} MISMATCH(ES):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll checked claims reproduced within tolerance (+/-0.02). PASS.")
        sys.exit(0)


if __name__ == "__main__":
    main()
