# =============================================================================
# MAZU Saudi Arabia — Structural Knowledge Graph (API-free)
#
# Builds a real multi-relation knowledge graph from:
#   1. Indicator metadata      (formula, unit, meaning, data source)
#   2. Hazard detection logic   (flash_flood_risk formula -> contributes_to)
#   3. Domain mechanisms        (ARST, moisture transport... -> triggers)
#   4. Real 2025 events         (6 auto-detected annual-extreme instances,
#                                 plus the 12 site-verified events from
#                                 deploy/index.html's map-verification section)
#   5. Data-driven correlations (computed from mazu_dataset.nc)
#
# Node types : Indicator, Hazard, DataSource, Region, Mechanism, Event
# Edge types : sourced_from, contributes_to, triggers, occurs_at, correlates_with
#
# Output: kg/kg_data.json  (node-link JSON for the web dashboard)
# =============================================================================

import os
import json
import itertools
import numpy as np
import xarray as xr
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "data", "mazu_dataset.nc")
OUT_JSON = os.path.join(HERE, "kg_data.json")

# =============================================================================
# 1. NODES
# =============================================================================

# ── Data sources ─────────────────────────────────────────────────────────
DATA_SOURCES = {
    "DS1":  "Global Atmospheric Reanalysis V1.5",
    "DS2":  "Global Surface Daily Analysis",
    "DS4":  "Global SST Fusion Analysis",
    "DS8":  "Global Climate Normals (1991-2020)",
    "DS10": "Satellite Precipitation Retrieval",
}

# ── Indicators: name -> (long name, unit, source, hazard-relevance) ───────
INDICATORS = {
    "daily_precip_total":      ("Total daily precipitation", "mm", "DS2"),
    "daily_convective_precip": ("Convective precipitation", "mm", "DS2"),
    "daily_large_scale_precip":("Large-scale precipitation", "mm", "DS2"),
    "t2m_c":                   ("2 m air temperature", "degC", "DS2"),
    "tmax_c":                  ("Daily maximum temperature", "degC", "DS2"),
    "tmin_c":                  ("Daily minimum temperature", "degC", "DS2"),
    "heat_index_c":            ("Heat index (apparent temp)", "degC", "DS2"),
    "vpd_kpa":                 ("Vapour pressure deficit", "kPa", "DS2"),
    "cape":                    ("Convective available potential energy", "J/kg", "DS1"),
    "pwat":                    ("Precipitable water", "kg/m2", "DS1"),
    "ivt":                     ("Integrated vapour transport", "kg/m/s", "DS1"),
    "wind850_speed":           ("850 hPa wind speed", "m/s", "DS1"),
    "wind_shear_850_200":      ("850-200 hPa wind shear", "m/s", "DS1"),
    "daily_precip_anomaly":    ("Daily precipitation anomaly", "mm", "DS8"),
    "t2m_anomaly_c":           ("2 m temperature anomaly", "degC", "DS8"),
    "tmax_anomaly_c":          ("Max temperature anomaly", "degC", "DS8"),
    "sst_celsius":             ("Sea surface temperature", "degC", "DS4"),
    "flash_flood_risk":        ("Flash-flood risk score", "score", "derived"),
    "heatwave_day_flag":       ("Heatwave day flag", "flag", "DS8"),
    "heatwave_duration_days":  ("Heatwave duration", "days", "DS8"),
    "wind10_speed":            ("10 m wind speed", "m/s", "DS1"),
    "dewpoint_depression_c":   ("Dewpoint depression (dryness of air)", "degC", "DS2"),
}

# ── Hazards ──────────────────────────────────────────────────────────────
HAZARDS = {
    "flash_flood": "Flash Flood / Wadi Flooding",
    "heatwave":    "Extreme Heat / Heatwave",
    "dust_storm":        "Dust Storm",
    "coastal":     "Coastal / Marine Risk",
}
HAZARD_DESC_OVERRIDE = {
    "dust_storm": "Dust Storm (wind-lifted, per Shamal-driven mechanism)",
}

# ── Regions: cities + geographic features (lat, lon, kind) ───────────────
REGIONS = {
    "Jeddah":       (21.5, 39.2, "city"),
    "Mecca":        (21.4, 39.8, "city"),
    "Riyadh":       (24.7, 46.7, "city"),
    "Jizan":        (16.9, 42.6, "city"),
    "Dammam":       (26.4, 50.1, "city"),
    "Taif":         (21.3, 40.4, "city"),
    "Medina":       (24.5, 39.6, "city"),
    "Abha":         (18.2, 42.5, "city"),
    "Red Sea":      (20.0, 38.5, "sea"),
    "Persian Gulf": (26.5, 51.5, "sea"),
    "Arabian Sea":  (15.5, 55.0, "sea"),
    "Empty Quarter":(19.5, 52.0, "desert"),
    # Added for the 12 site-verified events below (not among the original 8
    # modeled cities, but real locations named in their captions).
    "Hail":         (27.52, 41.70, "city"),
    "Buraidah":     (26.33, 43.97, "city"),
}

# region -> hazards it is exposed to (domain knowledge)
REGION_HAZARD = {
    "Jeddah": ["flash_flood", "coastal", "heatwave"], "Mecca": ["flash_flood", "heatwave"],
    "Taif": ["flash_flood"], "Jizan": ["flash_flood", "coastal"], "Abha": ["flash_flood"],
    "Riyadh": ["heatwave", "dust_storm"], "Medina": ["heatwave", "dust_storm"],
    "Dammam": ["heatwave", "coastal", "dust_storm"], "Empty Quarter": ["heatwave", "dust_storm"],
    "Red Sea": ["coastal"], "Persian Gulf": ["coastal"], "Arabian Sea": ["coastal"],
}
# region -> mechanisms it is exposed to
REGION_MECH = {
    "Jeddah": ["moisture_transport", "ARST"], "Mecca": ["orographic_lift", "ARST"],
    "Taif": ["orographic_lift"], "Abha": ["orographic_lift"], "Jizan": ["moisture_transport", "orographic_lift"],
    "Riyadh": ["subtropical_high", "thermal_low"], "Medina": ["thermal_low"],
    "Dammam": ["moisture_transport", "thermal_low"], "Empty Quarter": ["subtropical_high", "thermal_low"],
    "Red Sea": ["moisture_transport"], "Persian Gulf": ["moisture_transport"], "Arabian Sea": ["moisture_transport"],
}

# ── Mechanisms (domain knowledge) ────────────────────────────────────────
MECHANISMS = {
    "ARST":              "Active Red Sea Trough - low-level convergence over the Red Sea",
    "moisture_transport":"Red Sea / Arabian Sea moisture transport",
    "subtropical_high":  "Subtropical / continental high (heat dome)",
    "thermal_low":       "Arabian thermal low (desert heat low)",
    "orographic_lift":   "Orographic lifting over Hejaz / Asir mountains",
}

# ── Events are DETECTED from the data (defining var, hazard, short name) ──
EVENT_DEFS = [
    ("daily_precip_total", "flash_flood", "extreme rain",   "mm"),
    ("cape",               "flash_flood", "convective instability", "J/kg"),
    ("ivt",                "flash_flood", "vapour surge",    "kg/m/s"),
    ("tmax_c",             "heatwave",    "extreme heat",    "C"),
    ("heat_index_c",       "heatwave",    "heat stress",     "C"),
    ("wind10_speed",       "dust_storm",  "extreme wind",    "m/s"),
]
# indicators whose real value we attach to each event (driven_by, with value)
EVENT_DRIVERS = ["daily_precip_total", "cape", "ivt", "pwat", "tmax_c", "heat_index_c", "vpd_kpa",
                 "wind850_speed", "dewpoint_depression_c"]


def nearest_region(lat, lon):
    return min(REGIONS, key=lambda r: (REGIONS[r][0] - lat) ** 2 + (REGIONS[r][1] - lon) ** 2)


def detect_events():
    """Find the actual max-value cell/date for each event-defining variable,
    with the real driving indicator values at that exact cell."""
    if not os.path.exists(DATASET):
        return {}
    ds = xr.open_dataset(DATASET)
    lat = ds.latitude.values
    lon = ds.longitude.values
    events = {}
    for var, hazard, name, unit in EVENT_DEFS:
        a = ds[var].values
        ti, yi, xi = np.unravel_index(np.nanargmax(a), a.shape)
        date = str(ds.time.values[ti])[:10]
        la, lo = float(lat[yi]), float(lon[xi])
        region = nearest_region(la, lo)
        drivers = {}
        for d in EVENT_DRIVERS:
            v = float(ds[d].values[ti, yi, xi])
            if np.isfinite(v):
                drivers[d] = round(v, 1)
        eid = f"E_{date.replace('-','')}_{var}"
        events[eid] = {"label": f"{date[5:]} {name}", "date": date, "hazard": hazard,
                       "region": region, "peak_var": var, "peak_val": round(float(a[ti, yi, xi]), 1),
                       "unit": unit, "lat": round(la, 1), "lon": round(lo, 1), "drivers": drivers}
    ds.close()
    return events


# =============================================================================
# 4b. THE 12 SITE-VERIFIED REAL 2025 EVENTS (deploy/index.html "03 · Map
# Verification" section) -- distinct from the 6 auto-detected annual-extreme
# events above. Each entry's date/region/verdict is taken directly from the
# published caption text; dates for events where the caption gave a range but
# not a single peak day were independently re-derived from mazu_dataset.nc
# itself (documented per-event below), never guessed.
# =============================================================================

# (key, date, hazard, regions, label, verdict, note)
# date: the single most-representative day (see per-event note for how it was
#   pinned down); regions: list of Region node ids this event occurred at/near
#   (empty list = genuinely no city-level location, disclosed as such, not a
#   guess); verdict: exact text of the site's own verdict badge, or None where
#   the site published no badge (the 3 earliest-verified events).
VERIFIED_EVENTS = [
    ("dammam_dust", "2025-05-17", "dust_storm", ["Dammam"],
     "Dust storm (Dammam), 16-19 May", None,
     "REAL and PREDICTED both show a Gulf-coast band at Dammam on 17-18 May; "
     "date pinned to 17 May, the window's actual wind10_speed/wind850_speed peak "
     "(8.5 m/s / 13.9 m/s at Dammam, independently confirmed from mazu_dataset.nc)."),
    ("dammam_heat", "2025-06-16", "heatwave", ["Dammam"],
     "Heatwave (Dammam), 11-14 June", None,
     "Site caption states Dammam peaked at 44.2C (below the 45C rule-engine "
     "threshold) but names no single day; independently re-derived from "
     "mazu_dataset.nc: 16 June is the actual peak (44.22C) -- one day after the "
     "caption's stated 11-14 June window and the rendered map's 10-15 June "
     "frame range, a real range/peak mismatch worth flagging, not smoothed over."),
    ("jeddah_flood", "2025-12-09", "flash_flood", ["Jeddah"],
     "Flash flood (Jeddah), 8-11 Dec", None,
     "Caption's own date (9 Dec is when a red cell appears in the raw grid), "
     "but that signal sits ~165km north of Jeddah itself -- a genuine, "
     "disclosed non-confirmation at the named city, kept for that reason."),
    ("haboob_dust", "2025-05-04", "dust_storm", [],
     "Dust storm (Haboob), 4-5 May", "Inconclusive",
     "National event with no single focal point (matches "
     "model/12_timeline_manifest.py's focus_points=None for this event); the "
     "real high-risk zone sits 400-800km away near the Sudan/Jordan/Iraq "
     "border, outside all 8 modeled cities -- no region edge added, since "
     "forcing one onto an existing city node would misstate the finding."),
    ("junjul_dust", "2025-06-30", "dust_storm", ["Riyadh"],
     "Dust storm, 30 Jun-5 Jul", "Strong HIT",
     "Riyadh, 30 June: real 1.00, model 0.99 -- the cleanest agreement in the "
     "whole 12-event verification set."),
    ("heatwave2_miss", "2025-06-30", "heatwave", ["Dammam"],
     "Heatwave, 2nd wave, 28 Jun-5 Jul", "MISS",
     "Dammam, 30 June: real rule score 0.55 (at threshold) but model only "
     "0.045 -- a genuine model miss the rule engine independently confirmed."),
    ("heatwave3", "2025-07-22", "heatwave", ["Mecca"],
     "Heatwave, 3rd wave, 20-27 Jul", "Partial hit",
     "Named cities (Dammam/Khobar/Al-Ahsa) missed -- an independent hit was "
     "found instead at Mecca: 22 Jul real 0.60/model 0.97."),
    ("heatwave4", "2025-08-04", "heatwave", ["Mecca"],
     "Heatwave, 4th wave, 29 Jul-5 Aug", "Partial hit",
     "Weak agreement at Dammam (both below threshold); a second, independent "
     "hit again at Mecca: 4 Aug real 0.60/model 0.90."),
    ("flood_jan", "2025-01-06", "flash_flood", ["Mecca"],
     "Flood, 6-7 Jan", "Inconclusive",
     "Mecca real score 1.00, but CAPE/IVT/PWAT are all NaN this date (known "
     "Jan-Mar data gap) and the date falls inside the model's training window "
     "-- an unreliable sample, honestly labeled rather than forced into HIT."),
    ("flood_mar", "2025-03-06", "flash_flood", ["Hail", "Buraidah"],
     "Flood, 6-7 Mar (Hail/Buraidah)", "MISS, plus a new finding",
     "Real score 0.0 throughout, but the model's predicted probability reaches "
     "0.94-0.98 at the same coordinates on 6-7 Mar -- an undisclosed model/rule "
     "disagreement. Same known Jan-Mar CAPE/IVT/PWAT data gap as flood_jan, so "
     "the MISS may partly reflect that gap rather than a clean model error."),
    ("flood_taif", "2025-08-14", "flash_flood", ["Taif"],
     "Flood, 14 Aug (Taif)", "Model/real disagreement",
     "14 Aug: real 0.15, model 0.51 -- model runs ahead of the real signal, "
     "possibly picking up terrain-related signal in Taif's mountainous setting."),
    ("flood_aug2728", "2025-08-26", "flash_flood", ["Abha", "Jizan"],
     "Flood, 27-28 Aug (Asir/Jizan)", "Mixed result",
     "Abha, 26 Aug: real 0.75/model 0.84 (hit); by 28-29 Aug model drops to "
     "0.37-0.44 while real stays 0.75 (missed persistence). Jizan, 26-27 Aug: "
     "real 0.60, model only 0.18-0.22 (miss)."),
]


def detect_verified_events():
    """Attach real driver-indicator values (same EVENT_DRIVERS list used for
    the 6 auto-detected events) to each of the 12 site-verified events, read
    at that event's own fixed date and region coordinates -- not an argmax
    search, since the date/location are already pinned by the site's own
    published verification, but the indicator VALUES themselves are always
    freshly read from mazu_dataset.nc, never copied from the caption text."""
    if not os.path.exists(DATASET):
        return {}
    ds = xr.open_dataset(DATASET)
    lat_full, lon_full = ds.latitude.values, ds.longitude.values
    times = np.array([str(t)[:10] for t in ds.time.values])
    out = {}
    for key, date, hazard, regions, label, verdict, note in VERIFIED_EVENTS:
        eid = f"EV_{key}"
        entry = {"label": label, "date": date, "hazard": hazard, "regions": regions,
                 "verdict": verdict, "note": note, "drivers": {}}
        if date in times and regions:
            ti = int(np.where(times == date)[0][0])
            primary = regions[0]
            plat, plon, _ = REGIONS[primary]
            yi = int(np.argmin(np.abs(lat_full - plat)))
            xi = int(np.argmin(np.abs(lon_full - plon)))
            for d in EVENT_DRIVERS:
                if d in ds.data_vars:
                    v = float(ds[d].values[ti, yi, xi])
                    if np.isfinite(v):
                        entry["drivers"][d] = round(v, 1)
        out[eid] = entry
    ds.close()
    return out


# hazard -> mechanisms that drive it (domain knowledge)
HAZARD_MECH = {
    "flash_flood": ["ARST", "moisture_transport", "orographic_lift"],
    "heatwave":    ["subtropical_high", "thermal_low"],
    "dust_storm":        ["thermal_low"],
    "coastal":     ["moisture_transport"],
}

# =============================================================================
# 2. HAND-ENCODED EDGES
# =============================================================================

# indicator -> hazard  (grounded in flash_flood_risk formula + physics)
CONTRIBUTES_TO = {
    "flash_flood": ["daily_precip_total", "daily_convective_precip", "cape",
                    "ivt", "pwat", "wind850_speed", "daily_precip_anomaly", "flash_flood_risk"],
    "heatwave":    ["tmax_c", "t2m_c", "heat_index_c", "vpd_kpa",
                    "tmax_anomaly_c", "t2m_anomaly_c", "heatwave_day_flag", "heatwave_duration_days"],
    "dust_storm":        ["wind850_speed", "wind_shear_850_200", "vpd_kpa", "wind10_speed", "dewpoint_depression_c"],
    "coastal":     ["sst_celsius", "wind850_speed", "ivt"],
}

# mechanism -> target (indicator or hazard)  triggers
TRIGGERS = {
    "ARST":               ["ivt", "pwat", "flash_flood"],
    "moisture_transport": ["pwat", "ivt", "sst_celsius"],
    "subtropical_high":   ["tmax_c", "heatwave"],
    "thermal_low":        ["tmax_c", "vpd_kpa", "dust_storm"],
    "orographic_lift":    ["daily_precip_total", "flash_flood"],
}

# =============================================================================
# 3. DATA-DRIVEN CORRELATIONS
# =============================================================================

def compute_correlations(threshold=0.6):
    """Region-mean daily time series -> pairwise Pearson correlation.
    Returns list of (a, b, r) for |r| >= threshold."""
    if not os.path.exists(DATASET):
        print("[WARN] dataset not found, skipping correlations")
        return []
    ds = xr.open_dataset(DATASET)
    inds = [v for v in INDICATORS if v in ds.data_vars]
    series = {}
    for v in inds:
        arr = ds[v].values  # (time, lat, lon)
        # daily region mean over valid cells
        ts = np.nanmean(arr.reshape(arr.shape[0], -1), axis=1)
        series[v] = ts
    ds.close()

    edges = []
    for a, b in itertools.combinations(inds, 2):
        x, y = series[a], series[b]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 30:
            continue
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        if np.isfinite(r) and abs(r) >= threshold:
            edges.append((a, b, round(float(r), 3)))
    return edges

# =============================================================================
# 4. BUILD GRAPH
# =============================================================================

def build():
    G = nx.DiGraph()
    events = detect_events()
    verified_events = detect_verified_events()

    # ── nodes ───────────────────────────────────────────────────────────
    for k, v in DATA_SOURCES.items():
        G.add_node(k, ntype="DataSource", label=k, desc=v)
    for k, (ln, unit, src) in INDICATORS.items():
        G.add_node(k, ntype="Indicator", label=k, desc=ln, unit=unit, source=src)
    for k, v in HAZARDS.items():
        # dust_storm keeps its own richer, previously-shipped desc (from the
        # earlier dust/dust_storm rename work, agent/02_test_tools.py's
        # cap_alert_tool check depends on the exact "Dust Storm" label);
        # every other hazard's desc mirrors its label as before.
        desc = HAZARD_DESC_OVERRIDE.get(k, v)
        G.add_node(k, ntype="Hazard", label=v, desc=desc)
    for k, (lat, lon, kind) in REGIONS.items():
        G.add_node(k, ntype="Region", label=k, lat=lat, lon=lon, kind=kind)
    for k, v in MECHANISMS.items():
        G.add_node(k, ntype="Mechanism", label=k.replace("_", " "), desc=v)
    for k, e in events.items():
        G.add_node(k, ntype="Event", label=e["label"], date=e["date"], hazard=e["hazard"],
                   value=f"{e['peak_var']} {e['peak_val']} {e['unit']}",
                   location=f"{e['region']} ({e['lat']}N,{e['lon']}E)")
    for k, e in verified_events.items():
        # location format must match the auto-detected events' "Name (LATN,LONE)"
        # pattern -- agent/tools.py's similar_events_tool parses this exact
        # shape for every Event node under a queried hazard, with no fallback.
        # haboob_dust has no city-level coordinates (a genuinely national
        # event, disclosed as such); giving it a real point here would
        # misstate the finding, so its location deliberately does NOT match
        # that pattern -- similar_events_tool is made robust to that instead
        # of forcing a fake coordinate (see the try/except added there).
        # location string must be ONLY "Name (LATN,LONE)" -- no trailing text --
        # since _parse_event_location splits strictly on that pattern. Extra
        # locations for multi-city events (e.g. flood_mar's Hail + Buraidah)
        # go in the note/desc field instead, not appended here.
        if e["regions"]:
            primary = e["regions"][0]
            plat, plon, _ = REGIONS[primary]
            loc = f"{primary} ({plat}N,{plon}E)"
        else:
            loc = "outside 8-city coverage (national event, no fixed coordinates)"
        verdict_str = e["verdict"] or "no verdict badge published"
        G.add_node(k, ntype="Event", label=e["label"], date=e["date"], hazard=e["hazard"],
                   location=loc, verdict=verdict_str, value=f"verdict: {verdict_str}",
                   desc=e["note"], source="site-verified (deploy/index.html)")

    # ── sourced_from : indicator -> datasource ──────────────────────────
    for ind, (_, _, src) in INDICATORS.items():
        if src in DATA_SOURCES:
            G.add_edge(ind, src, etype="sourced_from")

    # ── contributes_to : indicator -> hazard ────────────────────────────
    for hz, inds in CONTRIBUTES_TO.items():
        for ind in inds:
            if ind in INDICATORS:
                G.add_edge(ind, hz, etype="contributes_to")

    # ── triggers : mechanism -> indicator/hazard ────────────────────────
    for mech, targets in TRIGGERS.items():
        for t in targets:
            if t in INDICATORS or t in HAZARDS:
                G.add_edge(mech, t, etype="triggers")

    # ── driven_by : hazard -> mechanism ─────────────────────────────────
    for hz, mechs in HAZARD_MECH.items():
        for m in mechs:
            if m in MECHANISMS:
                G.add_edge(hz, m, etype="driven_by")

    # ── at_risk_of : region -> hazard ───────────────────────────────────
    for reg, hzs in REGION_HAZARD.items():
        for hz in hzs:
            if hz in HAZARDS:
                G.add_edge(reg, hz, etype="at_risk_of")

    # ── exposed_to : region -> mechanism ────────────────────────────────
    for reg, mechs in REGION_MECH.items():
        for m in mechs:
            if m in MECHANISMS:
                G.add_edge(reg, m, etype="exposed_to")

    # ── event edges : manifests_as (hazard), occurs_at (region),
    #                  driven_by_value (indicator, with real value) ──────
    for k, e in events.items():
        G.add_edge(k, e["hazard"], etype="manifests_as")
        if e["region"] in REGIONS:
            G.add_edge(k, e["region"], etype="occurs_at")
        # attach the strongest real drivers as valued edges
        for ind, val in sorted(e["drivers"].items(), key=lambda x: -x[1])[:4]:
            if ind in INDICATORS:
                G.add_edge(k, ind, etype="observed_value", value=val)

    # ── verified-event edges (same pattern as the 6 auto-detected events,
    #    plus multiple occurs_at edges where a caption names >1 location) ──
    for k, e in verified_events.items():
        G.add_edge(k, e["hazard"], etype="manifests_as")
        for reg in e["regions"]:
            if reg in REGIONS:
                G.add_edge(k, reg, etype="occurs_at")
        for ind, val in sorted(e["drivers"].items(), key=lambda x: -x[1])[:4]:
            if ind in INDICATORS:
                G.add_edge(k, ind, etype="observed_value", value=val)

    # ── correlates_with : indicator <-> indicator (data-driven) ─────────
    corr = compute_correlations(threshold=0.6)
    for a, b, r in corr:
        G.add_edge(a, b, etype="correlates_with", weight=r)

    return G, corr


def main():
    G, corr = build()

    # node-link JSON for the dashboard
    data = nx.node_link_data(G, edges="links")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    # summary
    from collections import Counter
    ntypes = Counter(nx.get_node_attributes(G, "ntype").values())
    etypes = Counter(nx.get_edge_attributes(G, "etype").values())
    print("=" * 55)
    print("MAZU Structural Knowledge Graph")
    print("=" * 55)
    print(f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")
    print("\nNode types:")
    for k, v in ntypes.items():
        print(f"  {k:12s}: {v}")
    print("\nEdge types:")
    for k, v in etypes.items():
        print(f"  {k:16s}: {v}")
    print(f"\nData-driven correlations (|r|>=0.6): {len(corr)}")
    print("  Top 8:")
    for a, b, r in sorted(corr, key=lambda x: -abs(x[2]))[:8]:
        print(f"    {a:24s} <-> {b:24s} r={r:+.2f}")
    print(f"\n[SAVED] {OUT_JSON}")


if __name__ == "__main__":
    main()
