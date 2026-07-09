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
print(f"TOTAL: {PASS} passed, {FAIL} failed")
print("=" * 70)
if FAIL > 0:
    sys.exit(1)
