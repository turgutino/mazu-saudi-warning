# =============================================================================
# MAZU Saudi Arabia — Layer 1: Baseline Detection Engine
#
# Weighted multi-condition rules + spatial connected-component clustering,
# grounded in data-derived percentile thresholds and the teacher's report.
#
# For a given date it produces, per hazard:
#   - a continuous risk field (0..1) over the 160x220 grid
#   - discrete Event clusters (region, severity, peak risk, met conditions)
#
# This matches the competitor's rule-based detection (their ceiling); Layers
# 2-4 (STGNN forecast, causal KG, agent) then go beyond it.
# =============================================================================

import os
import json
import numpy as np
import xarray as xr
from scipy.ndimage import label as cc_label
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "data", "mazu_dataset.nc")

# ── Cities / features for region naming ──────────────────────────────────
REGIONS = {
    "Jeddah": (21.5, 39.2), "Mecca": (21.4, 39.8), "Riyadh": (24.7, 46.7),
    "Jizan": (16.9, 42.6), "Dammam": (26.4, 50.1), "Taif": (21.3, 40.4),
    "Medina": (24.5, 39.6), "Abha": (18.2, 42.5),
    "Red Sea": (20.0, 38.5), "Persian Gulf": (26.5, 51.5),
    "Arabian Sea": (15.5, 55.0), "Empty Quarter": (19.5, 52.0),
    "Northern Border": (31.0, 42.0),   # An Nafud desert / Iraq-Jordan border area,
                                        # where the dust-storm rule's known 2025-06-19
                                        # peak cell (31.9N,44.3E) falls -- added when
                                        # the dust_storm hazard was introduced.
}

# ── Rules: data-grounded thresholds (percentiles verified from dataset) ───
# weight sums to 1.0 per rule; "primary" is the dominant condition.
# Flash flood is ANCHORED on observed heavy rain (daily_precip_total>=10 is
# only ~0.49% of grid-days -> genuinely rare). Supporting convective/moisture
# conditions raise confidence. risk_threshold 0.5 requires the anchor plus
# support, so broad "favourable but dry" areas are NOT flagged as events.
RULES = {
    "flash_flood": {
        "conditions": [
            {"ind": "daily_precip_total", "op": ">=", "thr": 10,   "w": 0.40, "primary": True},  # observed rain, ~p99.5
            {"ind": "flash_flood_risk",   "op": ">=", "thr": 2,    "w": 0.20},   # teacher composite
            {"ind": "cape",               "op": ">=", "thr": 1000, "w": 0.15},   # ~p92 instability
            {"ind": "ivt",                "op": ">=", "thr": 200,  "w": 0.15},   # ~p95 moisture transport
            {"ind": "pwat",               "op": ">=", "thr": 40,   "w": 0.10},   # ~p97 moisture
        ],
        "min_cluster": 3, "connectivity": 4, "risk_threshold": 0.5,
        "severity": [("low", 0.0), ("medium", 0.5), ("high", 0.7), ("extreme", 0.85)],
    },
    "heatwave": {
        "conditions": [
            {"ind": "tmax_c",                 "op": ">=", "thr": 45, "w": 0.35, "primary": True},  # ~p95 observed heat
            {"ind": "heatwave_day_flag",      "op": ">=", "thr": 1,  "w": 0.25},
            {"ind": "heatwave_duration_days", "op": ">=", "thr": 3,  "w": 0.20},
            {"ind": "heat_index_c",           "op": ">=", "thr": 40, "w": 0.20},   # ~p91
        ],
        "min_cluster": 8, "connectivity": 8, "risk_threshold": 0.55,
        "severity": [("caution", 0.0), ("warning", 0.55), ("alert", 0.7), ("emergency", 0.85)],
    },
    # Dust storms need BOTH strong wind (surface + low-level jet, capable of
    # lifting loose material) AND dry air (loose, liftable soil; humid coastal
    # wind alone does not produce dust) -- primary condition is surface wind,
    # since that is the direct lifting mechanism; the other three are
    # data-grounded percentile thresholds (~p93-p95, verified against the
    # dataset in the conversation that added this hazard, June-August 2025 --
    # matching the KG's already-cited Shamal season, Yu et al. 2016).
    "dust_storm": {
        "conditions": [
            {"ind": "wind10_speed",         "op": ">=", "thr": 7.0,  "w": 0.35, "primary": True},  # ~p95
            {"ind": "wind850_speed",        "op": ">=", "thr": 11.0, "w": 0.25},  # ~p95, low-level jet
            {"ind": "dewpoint_depression_c","op": ">=", "thr": 38.0, "w": 0.20},  # ~p95, dry/liftable soil
            {"ind": "vpd_kpa",              "op": ">=", "thr": 5.5,  "w": 0.20},  # ~p93, dry atmosphere
        ],
        "min_cluster": 8, "connectivity": 8, "risk_threshold": 0.55,
        "severity": [("caution", 0.0), ("warning", 0.55), ("alert", 0.7), ("emergency", 0.85)],
    },
}

_OPS = {">=": np.greater_equal, ">": np.greater, "<=": np.less_equal, "<": np.less}


class DetectionEngine:
    def __init__(self, dataset=DATASET):
        self.ds = xr.open_dataset(dataset)
        self.lat = self.ds.latitude.values
        self.lon = self.ds.longitude.values
        self.times = np.array([str(t)[:10] for t in self.ds.time.values])

    # ── risk field: continuous 0..1 weighted score per grid cell ─────────
    def risk_field(self, date, hazard):
        rule = RULES[hazard]
        ti = int(np.where(self.times == date)[0][0])
        score = np.zeros((len(self.lat), len(self.lon)), dtype="float32")
        wsum = np.zeros_like(score)                      # available-weight sum (fallback aware)
        for c in rule["conditions"]:
            if c["ind"] not in self.ds:
                continue
            a = self.ds[c["ind"]].values[ti]
            valid = np.isfinite(a)
            hit = np.zeros_like(score)
            hit[valid] = _OPS[c["op"]](a[valid], c["thr"]).astype("float32")
            score += hit * c["w"]
            wsum += valid.astype("float32") * c["w"]
        # normalise by available weight (so missing indicators don't penalise)
        risk = np.where(wsum > 0, score / wsum, 0.0)
        return risk.astype("float32")

    def _severity(self, val, levels):
        lab = levels[0][0]
        for name, lo in levels:
            if val >= lo:
                lab = name
        return lab

    def _region(self, la, lo):
        r = min(REGIONS, key=lambda k: (REGIONS[k][0] - la) ** 2 + (REGIONS[k][1] - lo) ** 2)
        dist = ((REGIONS[r][0] - la) ** 2 + (REGIONS[r][1] - lo) ** 2) ** 0.5
        return r if dist <= 2.5 else f"{r} area"   # >2.5deg from any named point

    # ── detect: cluster the risk field into discrete events ──────────────
    def detect(self, date, hazard, risk_threshold=None):
        rule = RULES[hazard]
        if risk_threshold is None:
            risk_threshold = rule.get("risk_threshold", 0.5)
        risk = self.risk_field(date, hazard)
        mask = risk >= risk_threshold
        struct = np.ones((3, 3)) if rule["connectivity"] == 8 else \
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        clusters, n = cc_label(mask, structure=struct)
        events = []
        for cid in range(1, n + 1):
            ys, xs = np.where(clusters == cid)
            if len(ys) < rule["min_cluster"]:
                continue
            peak = int(np.argmax(risk[ys, xs]))
            pyi, pxi = ys[peak], xs[peak]
            la, lo = float(self.lat[pyi]), float(self.lon[pxi])
            peak_risk = float(risk[pyi, pxi])
            # which conditions fired at the peak cell
            fired = []
            for c in rule["conditions"]:
                if c["ind"] in self.ds:
                    v = float(self.ds[c["ind"]].values[
                        int(np.where(self.times == date)[0][0]), pyi, pxi])
                    if np.isfinite(v) and _OPS[c["op"]](v, c["thr"]):
                        fired.append(f"{c['ind']}={v:.1f}")
            events.append({
                "hazard": hazard, "date": date,
                "region": self._region(la, lo), "lat": round(la, 2), "lon": round(lo, 2),
                "cluster_size": int(len(ys)),
                "peak_risk": round(peak_risk, 3),
                "severity": self._severity(peak_risk, rule["severity"]),
                "conditions": fired,
            })
        events.sort(key=lambda e: -e["peak_risk"])
        return events

    def explain(self, e):
        return (f"[{e['severity'].upper()}] {e['hazard']} near {e['region']} "
                f"({e['lat']}N,{e['lon']}E) on {e['date']}: risk {e['peak_risk']:.2f}, "
                f"cluster {e['cluster_size']} cells. Drivers: {', '.join(e['conditions'])}")

    def close(self):
        self.ds.close()


# =============================================================================
# TEST on known 2025 extreme events
# =============================================================================
if __name__ == "__main__":
    eng = DetectionEngine()
    print("=" * 66)
    print("MAZU Detection Engine — validation on known 2025 events")
    print("=" * 66)

    tests = [
        ("2025-08-23", "flash_flood", "Jizan extreme rain 254.9mm"),
        ("2025-08-19", "flash_flood", "Arabian Sea IVT surge 728"),
        ("2025-07-25", "heatwave",    "Empty Quarter Tmax 53.7C"),
        ("2025-08-16", "heatwave",    "Persian Gulf heat-index 54.7C"),
        ("2025-06-19", "dust_storm",  "Northern Border wind+dry peak (found via grid search, "
                                       "falls within the KG's cited May-Aug Shamal season)"),
    ]
    for date, hz, note in tests:
        evs = eng.detect(date, hz)
        print(f"\n### {date} {hz}  ({note})")
        print(f"    detected {len(evs)} cluster(s)")
        for e in evs[:3]:
            print("    " + eng.explain(e))

    # negative control: a calm winter day should give few/no clusters
    print("\n" + "=" * 66)
    print("Negative control (calm days — should be quiet)")
    for date in ["2025-02-10", "2025-11-05"]:
        ff = eng.detect(date, "flash_flood")
        hw = eng.detect(date, "heatwave")
        ds_ = eng.detect(date, "dust_storm")
        print(f"  {date}: flash_flood={len(ff)} cluster(s), heatwave={len(hw)} cluster(s), "
              f"dust_storm={len(ds_)} cluster(s)")

    # annual summary stratified by severity (the honest picture)
    print("\n" + "=" * 66)
    print("Annual detection by severity (2025) — days with >=1 cluster of that tier")
    for hz in RULES:
        tiers = {}
        for d in eng.times:
            sev_here = set(e["severity"] for e in eng.detect(d, hz))
            for s in sev_here:
                tiers[s] = tiers.get(s, 0) + 1
        order = [t[0] for t in RULES[hz]["severity"]]
        print(f"  {hz}:")
        for s in order:
            print(f"      {s:10s}: {tiers.get(s,0):3d} days")
    eng.close()
