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
print("EXTENSION: dust_storm as a 3rd hazard, across all 4 tools")
print("=" * 70)

# --- forecast_tool: label was built from Layer 1's already-validated
# DetectionEngine RULES['dust_storm'] (not a separately-invented definition),
# model trained t->t+1 with the SAME methodology as flash_flood/heatwave
# (Jan-Jun train, Jul-Dec test). ROC-AUC/PR-AUC verified in
# model/dust_storm_forecast_report.txt.
r_dust = tools.forecast_tool("Riyadh", "2025-07-06", "dust_storm")
print(" ", "Riyadh 2025-07-06 dust_storm:", {k: v for k, v in r_dust.items() if k != "reflexive_check"})
check("dust_storm: no error", "error" not in r_dust, r_dust)
check("dust_storm: probability in [0,1]", 0.0 <= r_dust.get("probability", -1) <= 1.0)
check("dust_storm: reports its own verified ROC-AUC (0.8866, distinct from flash_flood/heatwave)",
     abs(r_dust.get("model_verified_roc_auc", 0) - 0.8866) < 0.001, r_dust.get("model_verified_roc_auc"))
check("dust_storm: reflexive_check present (reuses the same generic mechanism)",
     r_dust.get("reflexive_check") is not None, r_dust.get("reflexive_check"))

# Real, found-then-independently-verified elevated case: Dammam around the
# known dust period (06-19 to 07-07) shows BOTH signals agreeing elevated --
# found by scanning all 8 cities across several dates, then independently
# re-derived from the raw source file (2025-07-05, the features day for a
# 07-06 forecast): wind10_speed=9.01 (>=7, fires), wind850_speed=18.40
# (>=11, fires), dewpoint_depression_c=23.53 (<38, does NOT fire),
# vpd_kpa=4.54 (<5.5, does NOT fire) -> 0.35+0.25=0.60, matching exactly.
r_dammam = tools.forecast_tool("Dammam", "2025-07-06", "dust_storm")
rc_dammam = r_dammam.get("reflexive_check")
print(" ", "Dammam 2025-07-06 dust_storm reflexive_check:", rc_dammam)
check("Dammam 07-06: consistency = consistent_elevated (found + independently verified from raw source)",
     rc_dammam is not None and rc_dammam["consistency"] == "consistent_elevated", rc_dammam)
check("Dammam 07-06: detection score matches independent hand-computation from raw values (0.60)",
     rc_dammam is not None and abs(rc_dammam["detection_engine_risk_score"] - 0.60) < 0.01, rc_dammam)

# Negative control: a calm winter day should show low dust_storm probability
# and consistent_low, same pattern as the other 2 hazards.
r_calm_dust = tools.forecast_tool("Riyadh", "2025-01-15", "dust_storm")
check("dust_storm calm day: low probability", r_calm_dust.get("probability", 1) < 0.3, r_calm_dust.get("probability"))
check("dust_storm calm day: consistent_low",
     r_calm_dust.get("reflexive_check", {}).get("consistency") == "consistent_low", r_calm_dust)

# --- causal_kg_tool: dust_storm hazard node (renamed from the KG's original
# 'dust' id during this extension) must resolve, with 'thermal_low' as its
# mechanism -- ALREADY literature-grounded via the existing Yu et al. (2016)
# Shamal citation (that citation's own evidence quotes literally mention
# "capable of lifting dust and transporting it to the Persian Gulf and the
# Arabian Peninsula" -- grounding it for dust_storm required zero new
# literature extraction, only correcting the id/edges).
k_dust = tools.causal_kg_tool("dust_storm")
print(" ", "dust_storm mechanisms:", [m["mechanism"] for m in k_dust.get("mechanisms", [])])
check("dust_storm: no error", "error" not in k_dust, k_dust)
check("dust_storm: thermal_low mechanism present", "thermal_low" in [m["mechanism"] for m in k_dust["mechanisms"]])
thermal_low_entry = next(m for m in k_dust["mechanisms"] if m["mechanism"] == "thermal_low")
check("dust_storm: thermal_low is literature-grounded (reused Shamal citation, not fabricated)",
     thermal_low_entry["literature_grounded"] is True, thermal_low_entry)
check("dust_storm: citation is the Yu et al. 2016 Shamal paper",
     any("Yu et al" in c["citation"] for c in thermal_low_entry["citations"]), thermal_low_entry["citations"])
check("dust_storm: contributing_indicators includes the 2 newly-added raw variables",
     "wind10_speed" in k_dust["contributing_indicators"] and "dewpoint_depression_c" in k_dust["contributing_indicators"],
     k_dust["contributing_indicators"])

# --- similar_events_tool: 1 real dust_storm event exists in the KG (auto-
# detected the same way as the other 5: annual grid-max of the headline
# variable, wind10_speed, on 2025-07-26 near the Arabian Sea).
s_dust = tools.similar_events_tool("Dammam", "2025-07-06", "dust_storm")
check("dust_storm similar_events: no error", "error" not in s_dust, s_dust)
check("dust_storm similar_events: exactly 1 event compared against (only 1 dust_storm event in KG)",
     len(s_dust["ranked_similar_events"]) + len(s_dust["excluded_events"]) == 1,
     (s_dust["ranked_similar_events"], s_dust["excluded_events"]))

# Error handling for the new hazard value across all 3 hazard-taking tools.
check("dust_storm typo -> error (forecast_tool)", "error" in tools.forecast_tool("Riyadh", "2025-07-06", "duststorm"))
check("dust_storm valid in causal_kg_tool's own hazard enum (no KeyError)", "error" not in tools.causal_kg_tool("dust_storm"))

print()
print("=" * 70)
print("TOOL 5: region_risk_tool")
print("=" * 70)

# Jizan is at_risk_of exactly 2 hazards (flash_flood, coastal) per the KG's
# hand-encoded REGION_HAZARD -- independently re-checked by reading the KG
# JSON directly (bypassing the tool) rather than trusting its own count.
with open(tools.KG_JSON, encoding="utf-8") as _f:
    _kg = __import__("json").load(_f)
_jizan_haz_direct = sorted(l["target"] for l in _kg["links"]
                           if l.get("etype") == "at_risk_of" and l["source"] == "Jizan")

r_jizan = tools.region_risk_tool("Jizan")
print(" ", "Jizan region_risk (no date):", r_jizan)
check("Jizan: no error", "error" not in r_jizan, r_jizan)
check("Jizan: hazard list matches KG JSON read directly (bypass test)",
     sorted(h["hazard"] for h in r_jizan["hazards"]) == _jizan_haz_direct,
     (sorted(h["hazard"] for h in r_jizan["hazards"]), _jizan_haz_direct))
check("Jizan: flash_flood entry has_forecast_model=True",
     next(h for h in r_jizan["hazards"] if h["hazard"] == "flash_flood")["has_forecast_model"] is True)
check("Jizan: coastal entry has_forecast_model=False (KG hazard with no trained model, "
     "handled gracefully not errored)",
     next(h for h in r_jizan["hazards"] if h["hazard"] == "coastal")["has_forecast_model"] is False)
check("Jizan: flash_flood's city-specific mechanisms are a SUBSET of its full mechanism list "
     "(moisture_transport+orographic_lift subset of ARST+moisture_transport+orographic_lift)",
     set(next(h for h in r_jizan["hazards"] if h["hazard"] == "flash_flood")["mechanisms_affecting_this_city"])
     <= set(next(h for h in r_jizan["hazards"] if h["hazard"] == "flash_flood")["all_mechanisms_for_this_hazard"]))
check("Jizan: no 'date' key when date not requested (no accidental None-date pollution)",
     "date" not in r_jizan, r_jizan)

# Real finding (investigated, confirmed pre-existing, not introduced by this
# tool): Jeddah's REGION_HAZARD (01_build_structural_kg.py) lists it at_risk_of
# heatwave, but its REGION_MECH only lists moisture_transport/ARST (flash-
# flood-relevant mechanisms) -- NOT subtropical_high/thermal_low (heatwave's
# actual drivers). This is a genuine, disclosed minor KG data-quality gap
# that predates this session; the tool's correct behavior is to report an
# EMPTY mechanisms_affecting_this_city list for that hazard (honest, not a
# crash or a fabricated mechanism), which is what this test locks in.
r_jeddah = tools.region_risk_tool("Jeddah")
jeddah_hw = next(h for h in r_jeddah["hazards"] if h["hazard"] == "heatwave")
check("Jeddah/heatwave: mechanisms_affecting_this_city is empty (real, pre-existing KG gap, "
     "correctly disclosed as empty rather than fabricated or crashing)",
     jeddah_hw["mechanisms_affecting_this_city"] == [], jeddah_hw)
check("Jeddah/heatwave: all_mechanisms_for_this_hazard is still correctly populated "
     "(the gap is city-specific exposure, not the hazard's own mechanism data)",
     set(jeddah_hw["all_mechanisms_for_this_hazard"]) == {"subtropical_high", "thermal_low"}, jeddah_hw)

# Riyadh WITH a date: forecast should be attached for both its hazards
# (heatwave, dust_storm -- both have trained models), matching independently
# re-called forecast_tool outputs exactly (not just "present", but the EXACT
# same probability -- confirms region_risk_tool isn't silently drifting from
# forecast_tool's own logic by reimplementing it).
r_riyadh = tools.region_risk_tool("Riyadh", "2025-07-06")
riyadh_hw_direct = tools.forecast_tool("Riyadh", "2025-07-06", "heatwave")
riyadh_dust_direct = tools.forecast_tool("Riyadh", "2025-07-06", "dust_storm")
riyadh_hw_entry = next(h for h in r_riyadh["hazards"] if h["hazard"] == "heatwave")
riyadh_dust_entry = next(h for h in r_riyadh["hazards"] if h["hazard"] == "dust_storm")
check("Riyadh+date: heatwave forecast probability matches a direct forecast_tool call exactly",
     riyadh_hw_entry["forecast"]["probability"] == riyadh_hw_direct["probability"],
     (riyadh_hw_entry["forecast"]["probability"], riyadh_hw_direct["probability"]))
check("Riyadh+date: dust_storm forecast probability matches a direct forecast_tool call exactly",
     riyadh_dust_entry["forecast"]["probability"] == riyadh_dust_direct["probability"],
     (riyadh_dust_entry["forecast"]["probability"], riyadh_dust_direct["probability"]))

# Bad date: error must be surfaced (date_error), not silently dropped, and
# hazards should still be listed (KG exposure info doesn't depend on date).
r_baddate = tools.region_risk_tool("Jeddah", "2099-01-01")
check("bad date: date_error field present and informative", "date_error" in r_baddate, r_baddate)
check("bad date: hazards still listed despite the date error (KG info doesn't need a valid date)",
     r_baddate["hazard_count"] == 3, r_baddate)
check("bad date: no hazard has a spurious 'forecast' key when the date failed",
     all("forecast" not in h for h in r_baddate["hazards"]), r_baddate)

# Every one of the 8 known cities must resolve with at least 1 hazard (no
# silent gaps in KG coverage for a city the agent otherwise fully supports).
for c in tools.CITIES:
    rc = tools.region_risk_tool(c)
    check(f"{c}: region_risk_tool resolves with >=1 hazard (no KG coverage gap)",
         "error" not in rc and rc.get("hazard_count", 0) >= 1, rc)

check("unknown city -> error", "error" in tools.region_risk_tool("Atlantis"))

# =============================================================================
# TOOL 6: cap_alert_tool -- CAP 1.2 XML generation
# =============================================================================
import xml.etree.ElementTree as ET

CAP_NS_MAP = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}

# High-probability, consistency-DISAGREEING day (independently confirmed via
# forecast_tool + _reflexive_check above this test file was written): model
# says 94%, but the independent rule-based engine's risk score is only 0.25
# (below its own 0.30 elevated threshold) -- a genuine model/rule mismatch,
# not a test bug. Exercises the "Possible" (not "Likely") certainty path.
r_cap1 = tools.cap_alert_tool("Jizan", "2025-03-27", "flash_flood")
check("cap_alert_tool: alert_warranted True for a high-probability day",
     r_cap1.get("alert_warranted") is True, r_cap1)
check("cap_alert_tool: probability matches an independent direct forecast_tool call",
     r_cap1["probability"] == tools.forecast_tool("Jizan", "2025-03-27", "flash_flood")["probability"],
     r_cap1)
check("cap_alert_tool: severity 'Extreme' for probability >= 0.85 (flash_flood's own threshold)",
     r_cap1["cap_severity"] == "Extreme", r_cap1)
check("cap_alert_tool: certainty 'Possible' when model/rule-engine DISAGREE (model_higher_than_detection)",
     r_cap1["cap_certainty"] == "Possible", r_cap1)

parsed1 = ET.fromstring(r_cap1["cap_xml"])
check("cap_alert_tool XML: root tag is CAP-namespaced <alert>",
     parsed1.tag == "{urn:oasis:names:tc:emergency:cap:1.2}alert", parsed1.tag)
check("cap_alert_tool XML: status is 'Exercise', never 'Actual' (honesty requirement -- "
     "this is a historical-dataset demo, not a live feed)",
     parsed1.find("cap:status", CAP_NS_MAP).text == "Exercise", r_cap1["cap_xml"])
check("cap_alert_tool XML: <severity> element matches the dict's cap_severity field exactly",
     parsed1.find("cap:info/cap:severity", CAP_NS_MAP).text == r_cap1["cap_severity"], r_cap1["cap_xml"])
check("cap_alert_tool XML: <identifier> embeds hazard, city, and target_date",
     parsed1.find("cap:identifier", CAP_NS_MAP).text == "MAZU-flash_flood-Jizan-2025-03-27",
     parsed1.find("cap:identifier", CAP_NS_MAP).text)
check("cap_alert_tool XML: <circle> area matches Jizan's known coordinates",
     parsed1.find("cap:info/cap:area/cap:circle", CAP_NS_MAP).text == "16.9,42.6 0",
     parsed1.find("cap:info/cap:area/cap:circle", CAP_NS_MAP).text)

# Consistency-AGREEING day (independently confirmed: dust_storm risk field at
# Dammam's grid cell exceeds the reflexive-check threshold too) -- exercises
# the "Likely" certainty path, and a 2nd hazard's CAP <event> label from the
# KG rather than the raw hazard id.
r_cap2 = tools.cap_alert_tool("Dammam", "2025-01-02", "dust_storm")
check("cap_alert_tool: certainty 'Likely' when model/rule-engine AGREE (consistent_elevated)",
     r_cap2["cap_certainty"] == "Likely", r_cap2)
parsed2 = ET.fromstring(r_cap2["cap_xml"])
check("cap_alert_tool XML: <event> uses the KG's human-readable label, not the raw hazard id",
     parsed2.find("cap:info/cap:event", CAP_NS_MAP).text == "Dust Storm", r_cap2["cap_xml"])

# Low-probability day: no alert should be issued at all (matches real warning
# systems -- not every day gets a CAP message), and no cap_xml key present.
r_cap3 = tools.cap_alert_tool("Jizan", "2025-08-20", "flash_flood")
check("cap_alert_tool: alert_warranted False for a sub-threshold probability day",
     r_cap3.get("alert_warranted") is False, r_cap3)
check("cap_alert_tool: no cap_xml key present when no alert is warranted",
     "cap_xml" not in r_cap3, r_cap3)
check("cap_alert_tool: reason field explains the threshold that wasn't met",
     "threshold" in r_cap3.get("reason", ""), r_cap3)

# Error propagation: bad city/date must surface the SAME error forecast_tool
# itself would produce (cap_alert_tool must not swallow or reword it).
check("cap_alert_tool: unknown city -> same error as forecast_tool",
     tools.cap_alert_tool("Atlantis", "2025-08-20", "flash_flood")
     == tools.forecast_tool("Atlantis", "2025-08-20", "flash_flood"))
check("cap_alert_tool: out-of-range date -> same error as forecast_tool",
     tools.cap_alert_tool("Jizan", "2099-01-01", "flash_flood")
     == tools.forecast_tool("Jizan", "2099-01-01", "flash_flood"))

# Every hazard must produce well-formed, re-parseable XML for a warranted
# alert (cross-hazard structural coverage, not just flash_flood). Each
# (city, date) below was independently confirmed above this line to clear
# its hazard's alert threshold -- asserted as a hard requirement, not
# silently skipped if a date happens to fall short.
for city, date, hz in [("Jizan", "2025-03-27", "flash_flood"),
                       ("Riyadh", "2025-08-27", "heatwave"),
                       ("Dammam", "2025-01-02", "dust_storm")]:
    rc = tools.cap_alert_tool(city, date, hz)
    check(f"cap_alert_tool: {hz}/{city} on {date} clears its own alert threshold "
         "(all 3 hazards must be covered by this loop, not silently skipped)",
         rc.get("alert_warranted") is True, rc)
    p = ET.fromstring(rc["cap_xml"])
    check(f"cap_alert_tool XML well-formed for {hz}/{city}",
         p.find("cap:info/cap:event", CAP_NS_MAP) is not None, rc["cap_xml"])

print()
print("=" * 70)
print(f"TOTAL: {PASS} passed, {FAIL} failed")
print("=" * 70)
if FAIL > 0:
    sys.exit(1)
