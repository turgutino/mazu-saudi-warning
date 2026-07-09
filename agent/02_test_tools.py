# =============================================================================
# MAZU — Layer 4: deep, independent verification of each tool BEFORE wiring
# them into the LLM agent. Matches the session's established pattern: verify
# every piece in isolation, with known events and negative/error cases.
#
# forecast_tool takes `target_date` -- the date whose risk is being forecast
# (it internally uses the PREVIOUS day's indicators). This interface was
# deliberately chosen after finding that an earlier "today+predicts tomorrow"
# design was an off-by-one trap for the LLM agent: a user asking "risk ON
# date X" naturally expects X to be the forecast target, not the input day.
# =============================================================================
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools

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


print("=" * 70)
print("TOOL 1: forecast_tool")
print("=" * 70)

# Known event: forecast risk ON 2025-08-23 (Jizan 254.9mm event), which should
# internally use 2025-08-22's indicators.
r = tools.forecast_tool("Jizan", "2025-08-23", "flash_flood")
print(" ", r)
check("no error", "error" not in r, r)
check("target_date echoed correctly", r.get("target_date") == "2025-08-23")
check("uses the PREVIOUS day's indicators", r.get("features_from_date") == "2025-08-22",
     f"got {r.get('features_from_date')}")
check("probability in [0,1]", 0.0 <= r.get("probability", -1) <= 1.0)
check("reports model ROC-AUC", r.get("model_verified_roc_auc") == tools._META["flash_flood"]["roc_auc"])

# Known event day 2025-07-25: checked conditions_tool first and found NONE of our
# 8 named cities hit the 45C extreme threshold that day (max was Riyadh 43.5C) --
# the true record (53.7C) was at a desert grid cell ("Empty Quarter") outside any
# named city.
#
# A naive test comparing RAW tmax ranking to forecast-probability ranking across
# all 8 cities gave only weak rank correlation (Spearman rho=0.38, not
# significant). Investigating why: our own heatwave_day_flag label (Layer 1) is
# defined by tmax ANOMALY relative to local climatology
# (tmax_c >= max(40, tmax_climatology_c + 5)), not an absolute cutoff. Checked
# conditions_tool for tmax_anomaly_c on the input date and confirmed: Mecca was
# +3.68C above its own climatology that day while Riyadh was -2.42C BELOW its
# climatology, despite Riyadh's higher absolute tmax -- so the model correctly
# assigning Mecca a higher heatwave probability reflects the anomaly-based
# physics of our own label definition, not a bug. The correct test is therefore
# anomaly-ranked, not raw-tmax-ranked.
r2a = tools.forecast_tool("Mecca", "2025-07-25", "heatwave")    # highest tmax_anomaly_c on the input day (+3.68C)
r2b = tools.forecast_tool("Abha", "2025-07-25", "heatwave")     # below-normal that day (-2.62C)
print(" ", r2a)
print(" ", r2b)
check("heatwave: no error", "error" not in r2a, r2a)
check("heatwave: target_date echoed correctly", r2a.get("target_date") == "2025-07-25")
check("heatwave: uses previous day's indicators", r2a.get("features_from_date") == "2025-07-24")
check("heatwave: city running ABOVE its own climatology (Mecca, +3.68C anomaly) "
     "ranks above a city running BELOW its climatology (Abha, -2.62C anomaly) "
     "-- consistent with our anomaly-based label definition",
     r2a.get("probability", 0) > r2b.get("probability", 1),
     f"Mecca={r2a.get('probability')} Abha={r2b.get('probability')}")

# Negative control: calm day should give LOW probability
r3 = tools.forecast_tool("Riyadh", "2025-11-06", "heatwave")
print(" ", r3)
check("calm day: low probability", r3.get("probability", 1) < 0.1, f"got {r3.get('probability')}")

# Error handling
r4 = tools.forecast_tool("Atlantis", "2025-08-23", "flash_flood")
check("unknown city -> error", "error" in r4, r4)
r5 = tools.forecast_tool("Jizan", "2099-01-01", "flash_flood")
check("out-of-range date -> error", "error" in r5, r5)
r6 = tools.forecast_tool("Jizan", "2025-01-01", "flash_flood")
check("first day (no prior day) -> error", "error" in r6, r6)
r7 = tools.forecast_tool("Jizan", "2025-08-23", "earthquake")
check("unknown hazard -> error", "error" in r7, r7)

print()
print("=" * 70)
print("EXTENSION: terrain/elevation context on forecast_tool")
print("=" * 70)

# Abha sits in the Asir mountains (~2082m per raw orography grid, verified
# against known real-world elevation ~2270m -- within grid-resolution error).
r8 = tools.forecast_tool("Abha", "2025-08-23", "heatwave")
print(" ", {k: v for k, v in r8.items() if k in ("elevation_m", "terrain_note")})
check("Abha: elevation_m is a real, high value (mountain city)",
     r8.get("elevation_m") is not None and r8["elevation_m"] > 1500, r8.get("elevation_m"))
check("Abha: terrain_note present (mountain caution flag)", r8.get("terrain_note") is not None)

# Jeddah and Dammam are coastal/near-sea-level -- should NOT get the mountain flag.
r9 = tools.forecast_tool("Jeddah", "2025-08-23", "heatwave")
r10 = tools.forecast_tool("Dammam", "2025-08-23", "heatwave")
print(" ", "Jeddah elevation_m:", r9.get("elevation_m"), "terrain_note:", r9.get("terrain_note"))
print(" ", "Dammam elevation_m:", r10.get("elevation_m"), "terrain_note:", r10.get("terrain_note"))
check("Jeddah: low elevation, no mountain flag",
     r9.get("elevation_m") is not None and r9["elevation_m"] < 500 and r9.get("terrain_note") is None,
     r9.get("elevation_m"))
check("Dammam: low elevation, no mountain flag",
     r10.get("elevation_m") is not None and r10["elevation_m"] < 500 and r10.get("terrain_note") is None,
     r10.get("elevation_m"))

# Taif is also elevated (~1774m) -- should also get the flag.
r11 = tools.forecast_tool("Taif", "2025-08-23", "heatwave")
check("Taif: elevated terrain, mountain flag present",
     r11.get("elevation_m") is not None and r11["elevation_m"] > 1500 and r11.get("terrain_note") is not None,
     r11.get("elevation_m"))

print()
print("=" * 70)
print("EXTENSION: reflexive_check (model vs. independent rule-based detection)")
print("=" * 70)

# Negative control: calm day should show BOTH signals low and agreeing.
r_calm = tools.forecast_tool("Riyadh", "2025-11-06", "heatwave")
rc = r_calm.get("reflexive_check")
print(" ", "Riyadh calm day:", rc)
check("calm day: reflexive_check present", rc is not None)
check("calm day: consistency = consistent_low",
     rc is not None and rc["consistency"] == "consistent_low", rc)
check("calm day: detection_engine_risk_score is low", rc is not None and rc["detection_engine_risk_score"] < 0.3)

# Jizan 2025-08-23 flash-flood: investigated in depth (see conversation) --
# the precursor day (08-22) had elevated CAPE/IVT/PWAT/composite risk but NOT
# yet observed rain (that started the 23rd), so the rule engine's supporting
# conditions fire (0.20+0.15+0.15+0.10=0.60) while its primary "actual rain"
# condition (weight 0.40) correctly does not. The model gave only 12.5%,
# which is LOWER than the physical preconditions would suggest -- a genuine,
# independently-confirmed finding (re-verified against raw indicator values
# directly, not just this tool's own output), not a bug in this check.
r_jizan = tools.forecast_tool("Jizan", "2025-08-23", "flash_flood")
rj = r_jizan.get("reflexive_check")
print(" ", "Jizan 08-23 flash_flood:", rj)
check("Jizan: reflexive_check present", rj is not None)
check("Jizan: consistency = detection_higher_than_model (known, investigated precursor-day case)",
     rj is not None and rj["consistency"] == "detection_higher_than_model", rj)
check("Jizan: detection score matches independently-verified raw-value computation "
     "(0.20 flash_flood_risk + 0.15 cape + 0.15 ivt + 0.10 pwat = 0.60, rain NOT yet fired)",
     rj is not None and abs(rj["detection_engine_risk_score"] - 0.60) < 0.01, rj)
check("Jizan: daily_precip_total did NOT fire (rain hadn't started on the precursor day)",
     rj is not None and not any(c.startswith("daily_precip_total") for c in rj["detection_engine_conditions_fired"]),
     rj.get("detection_engine_conditions_fired") if rj else None)

# Mecca 2025-07-25 heatwave: also investigated -- the precursor day (07-24)
# was anomalously hot for MECCA specifically (+3.68C above its own
# climatology) but did not cross the rule engine's ABSOLUTE thresholds
# (tmax_c>=45, heat_index_c>=40 -- actual values were 42.98C/37.0C). The ML
# model (trained on the anomaly-aware heatwave_day_flag label) correctly
# flags this as elevated risk (62.8%) where the simpler absolute-threshold
# rule engine sees nothing -- a real demonstration of the ML model adding
# value beyond fixed-threshold detection, not a flaw in either component.
r_mecca = tools.forecast_tool("Mecca", "2025-07-25", "heatwave")
rm = r_mecca.get("reflexive_check")
print(" ", "Mecca 07-25 heatwave:", rm)
check("Mecca: consistency = model_higher_than_detection (known, investigated anomaly-vs-absolute case)",
     rm is not None and rm["consistency"] == "model_higher_than_detection", rm)
check("Mecca: detection engine found zero fired conditions (absolute thresholds not met)",
     rm is not None and rm["detection_engine_risk_score"] == 0.0 and len(rm["detection_engine_conditions_fired"]) == 0,
     rm)

# The 4th consistency branch (consistent_elevated -- both signals agree risk
# IS elevated) had NOT been covered by a concrete example above. Found via a
# sampled search across all cities/hazards/dates, then independently
# verified against the raw source file BEFORE being hardcoded here (per the
# session's rule: never trust a found case without re-deriving it):
# Mecca 2025-05-17 heatwave, features from 2025-05-16 --
# raw tmax_c=43.39 (<45, does NOT fire), heat_index_c=35.17 (<40, does NOT
# fire), heatwave_day_flag=1.0 (fires, w=0.25), heatwave_duration_days=6.0
# (fires, w=0.20) -> independently hand-computed score = 0.25+0.20 = 0.45,
# matching the tool's own output exactly.
r_both = tools.forecast_tool("Mecca", "2025-05-17", "heatwave")
rb = r_both.get("reflexive_check")
print(" ", "Mecca 05-17 heatwave (consistent_elevated case):", rb)
check("Mecca 05-17: consistency = consistent_elevated (4th branch, found + independently verified)",
     rb is not None and rb["consistency"] == "consistent_elevated", rb)
check("Mecca 05-17: model probability actually elevated (>=0.3)",
     r_both["probability"] >= 0.3, r_both["probability"])
check("Mecca 05-17: detection score matches independent hand-computation from raw values (0.45)",
     rb is not None and abs(rb["detection_engine_risk_score"] - 0.45) < 0.01, rb)

# Bypass test: independently re-derive the detection risk score for the Jizan
# case directly from the DetectionEngine class (bypassing forecast_tool
# entirely) to confirm _reflexive_check() isn't silently drifting from the
# same rule engine Layer 1 already validated.
de_direct = tools.DetectionEngine(dataset=tools.DATASET)
direct_field = de_direct.risk_field("2025-08-22", "flash_flood")
yi_d = int((abs(de_direct.lat - 16.9)).argmin())
xi_d = int((abs(de_direct.lon - 42.6)).argmin())
direct_score = float(direct_field[yi_d, xi_d])
check("Jizan: reflexive_check score matches an independent, direct DetectionEngine call (bypass test)",
     rj is not None and abs(rj["detection_engine_risk_score"] - direct_score) < 1e-6,
     f"tool={rj['detection_engine_risk_score']} direct={direct_score}")
de_direct.close()

print()
print("=" * 70)
print("EXTENSION: impact_context (population reference) on forecast_tool")
print("=" * 70)

r12 = tools.forecast_tool("Riyadh", "2025-08-23", "heatwave")
ic = r12.get("impact_context")
print(" ", ic)
check("Riyadh: impact_context present", ic is not None)
check("Riyadh: population is the real 2022 census figure (~9.06M)",
     ic is not None and 8_000_000 < ic["city_population_2022_census"] < 10_000_000,
     ic.get("city_population_2022_census") if ic else None)
check("Riyadh: impact_context sourced (GASTAT)", ic is not None and "GASTAT" in ic["source"])
check("Riyadh: impact_context explicitly disclaims exposure modeling",
     ic is not None and "NOT" in ic["note"] and "exposure" in ic["note"])

r13 = tools.forecast_tool("Jizan", "2025-08-23", "flash_flood")
ic2 = r13.get("impact_context")
check("Jizan: population is real and plausible (~174k, smallest of the 8 cities)",
     ic2 is not None and 100_000 < ic2["city_population_2022_census"] < 300_000,
     ic2.get("city_population_2022_census") if ic2 else None)

# All 8 cities must resolve to SOME population figure -- a silent None would
# mean a city quietly loses this context with no error surfaced.
for c in tools.CITIES:
    rc = tools.forecast_tool(c, "2025-08-23", "heatwave")
    check(f"{c}: impact_context resolves (no missing population data)",
         rc.get("impact_context") is not None and
         isinstance(rc["impact_context"]["city_population_2022_census"], int),
         rc.get("impact_context"))

print()
print("=" * 70)
print("TOOL 2: causal_kg_tool")
print("=" * 70)

k1 = tools.causal_kg_tool("flash_flood")
print(f"  flash_flood mechanisms: {[m['mechanism'] for m in k1['mechanisms']]}")
check("flash_flood: no error", "error" not in k1, k1)
check("flash_flood: has mechanisms", len(k1["mechanisms"]) > 0)
check("flash_flood: mechanisms match KG design (ARST, moisture_transport, orographic_lift)",
     set(m["mechanism"] for m in k1["mechanisms"]) == {"ARST", "moisture_transport", "orographic_lift"},
     [m["mechanism"] for m in k1["mechanisms"]])
grounded = [m for m in k1["mechanisms"] if m["literature_grounded"]]
check("flash_flood: at least one mechanism literature-grounded", len(grounded) > 0,
     f"grounded={[m['mechanism'] for m in grounded]}")
if grounded:
    ev = grounded[0]["citations"][0]["evidence"]
    check("citation carries verbatim evidence quotes", len(ev) > 0 and "quote" in ev[0], ev)

k2 = tools.causal_kg_tool("heatwave")
print(f"  heatwave mechanisms: {[m['mechanism'] for m in k2['mechanisms']]}")
check("heatwave: no error", "error" not in k2, k2)
check("heatwave: mechanisms match KG design (subtropical_high, thermal_low)",
     set(m["mechanism"] for m in k2["mechanisms"]) == {"subtropical_high", "thermal_low"},
     [m["mechanism"] for m in k2["mechanisms"]])

k3 = tools.causal_kg_tool("volcano")
check("unknown hazard -> error", "error" in k3, k3)

print()
print("=" * 70)
print("TOOL 3: conditions_tool")
print("=" * 70)

c1 = tools.conditions_tool("Jizan", "2025-08-23")
print(f"  Jizan 2025-08-23: {c1.get('indicators', {}).get('daily_precip_total')} mm precip "
     f"(grid cell {c1.get('grid_cell')})")
check("conditions: no error", "error" not in c1, c1)
check("conditions: has precip value", "daily_precip_total" in c1.get("indicators", {}))
check("conditions: precip is a real observed number (not the global grid max, "
     "since this is the nearest cell to the CITY, not the storm centroid)",
     isinstance(c1["indicators"]["daily_precip_total"], float))

c2 = tools.conditions_tool("Riyadh", "2025-07-25")
print(f"  Riyadh 2025-07-25: tmax={c2.get('indicators', {}).get('tmax_c')}C")
check("conditions: Riyadh tmax is a hot, plausible value", c2["indicators"]["tmax_c"] > 35,
     c2["indicators"]["tmax_c"])

c3 = tools.conditions_tool("Atlantis", "2025-08-23")
check("unknown city -> error", "error" in c3, c3)

print()
print("=" * 70)
print("TOOL 4: similar_events_tool")
print("=" * 70)

# Core-math sanity check: querying AT the exact coordinates and date of a
# known event must score exactly 100% self-similarity (distance 0). This is
# the foundational check that the z-score/distance formula is implemented
# correctly, independent of any city-lookup complications.
_orig_jizan = tools.CITIES["Jizan"]
tools.CITIES["Jizan"] = (17.5, 42.9)   # the 08-23 event's own grid-max coords
s_self = tools.similar_events_tool("Jizan", "2025-08-23", "flash_flood")
self_match = next((x for x in s_self["ranked_similar_events"] if x["event"] == "08-23 extreme rain"), None)
check("self-match at exact event coords/date scores exactly 100%",
     self_match is not None and self_match["similarity_pct"] == 100.0, self_match)
check("self-match: event_distance_from_city_km is 0 when query IS the event location",
     self_match is not None and self_match["event_distance_from_city_km"] == 0, self_match)
tools.CITIES["Jizan"] = _orig_jizan   # restore -- must not leak into other tests

# Real-world case (investigated in conversation): Jizan CITY center on the
# 08-23 event's own date scores LOW similarity to that same-named event,
# because the event's coordinates are the storm's grid-max centroid, ~74km
# from the city center, where daily_precip_total was 0.6mm vs 254.9mm at the
# centroid -- independently confirmed against the raw source file. This is
# correct, disclosed behavior (hyperlocal extreme), not a bug.
s_city = tools.similar_events_tool("Jizan", "2025-08-23", "flash_flood")
same_day_event = next((x for x in s_city["ranked_similar_events"] if x["event"] == "08-23 extreme rain"), None)
check("Jizan city center vs the same-day/same-name event: LOW similarity (hyperlocal storm, correct behavior)",
     same_day_event is not None and same_day_event["similarity_pct"] < 5, same_day_event)
check("distance field correctly shows the city-to-centroid gap (~74km, independently computed)",
     same_day_event is not None and 65 <= same_day_event["event_distance_from_city_km"] <= 85,
     same_day_event)

# Real, found-then-independently-verified positive match: Mecca 2025-08-04
# was anomalously hot (raw tmax_c=45.95C, tmax_anomaly_c=+6.65C, independently
# read from the raw source file) -- genuinely similar in PROFILE (both
# anomalously hot for their own conditions) to the Empty Quarter's record
# 07-25 heat event (tmax_c=53.75C, anomaly +10.15C), despite the ~1659km
# distance and 8C absolute temperature gap -- the z-scored comparison
# correctly captures "both anomalous" rather than being fooled by the raw
# temperature difference.
s_mecca = tools.similar_events_tool("Mecca", "2025-08-04", "heatwave")
mecca_match = next((x for x in s_mecca["ranked_similar_events"] if x["event"] == "07-25 extreme heat"), None)
check("Mecca 08-04 vs 07-25 extreme-heat event: meaningfully high similarity (>=30%, found + independently verified)",
     mecca_match is not None and mecca_match["similarity_pct"] >= 30, mecca_match)
check("query_indicators echoes the real raw tmax_c (~45.95C, independently confirmed)",
     abs(s_mecca["query_indicators"]["tmax_c"] - 45.95) < 0.1, s_mecca["query_indicators"])

# Negative control: a calm day should NOT score highly against any event.
s_calm = tools.similar_events_tool("Riyadh", "2025-11-06", "heatwave")
check("calm day: no event scores above 30% similarity",
     all(x["similarity_pct"] < 30 for x in s_calm["ranked_similar_events"]),
     s_calm["ranked_similar_events"])

# Results must be sorted descending by similarity.
check("Mecca results are sorted descending by similarity_pct",
     all(s_mecca["ranked_similar_events"][i]["similarity_pct"] >= s_mecca["ranked_similar_events"][i + 1]["similarity_pct"]
         for i in range(len(s_mecca["ranked_similar_events"]) - 1)),
     [x["similarity_pct"] for x in s_mecca["ranked_similar_events"]])

# Structural / disclosure checks.
check("note field explicitly disclaims this is NOT a probability",
     "NOT a probability" in s_mecca["note"], s_mecca["note"])
check("flash_flood query only compares against flash_flood events (3 of them), not heatwave events",
     len(s_city["ranked_similar_events"]) + len(s_city["excluded_events"]) == 3,
     (s_city["ranked_similar_events"], s_city["excluded_events"]))
check("heatwave query only compares against heatwave events (2 of them)",
     len(s_mecca["ranked_similar_events"]) + len(s_mecca["excluded_events"]) == 2,
     (s_mecca["ranked_similar_events"], s_mecca["excluded_events"]))

# Bypass test: independently re-derive the Mecca/07-25 similarity score using
# raw values pulled directly and hand-computed z-score distance, bypassing
# similar_events_tool's internals (only reusing the already-tested
# _feature_stats cache for mean/std, which is itself dataset-derived, not
# hardcoded).
feats = tools.SIMILARITY_FEATURES["heatwave"]
import numpy as _np
mecca_raw = tools._get_vector("2025-08-04", *tools.CITIES["Mecca"], feats)
event_raw = tools._get_vector("2025-07-25", 18.7, 54.5, feats)
sq = []
for v in feats:
    mean, std = tools._feature_stats(v)
    sq.append(((mecca_raw[v] - mean) / std - (event_raw[v] - mean) / std) ** 2)
indep_dist = _np.sqrt(sum(sq))
indep_sim = round(100.0 / (1.0 + indep_dist), 1)
check("bypass test: independently hand-computed similarity matches tool output exactly",
     mecca_match is not None and abs(mecca_match["similarity_pct"] - indep_sim) < 0.05,
     f"tool={mecca_match['similarity_pct'] if mecca_match else None} independent={indep_sim}")

# Error handling.
check("unknown city -> error", "error" in tools.similar_events_tool("Atlantis", "2025-08-23", "flash_flood"))
check("unknown hazard -> error", "error" in tools.similar_events_tool("Jizan", "2025-08-23", "earthquake"))
check("out-of-range date -> error", "error" in tools.similar_events_tool("Jizan", "2099-01-01", "flash_flood"))

print()
print("=" * 70)
print(f"TOTAL: {PASS} passed, {FAIL} failed")
print("=" * 70)
if FAIL > 0:
    sys.exit(1)
