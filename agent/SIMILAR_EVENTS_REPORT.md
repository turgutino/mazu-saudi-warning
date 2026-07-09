# MAZU Extension — Similar Historical Events Tool

## What this is

A 4th agent tool, `similar_events_tool(city, date, hazard)`, compares a
city/date's actual raw indicators against the knowledge graph's 5 known real
2025 extreme events, returning a descriptive similarity score per event. This
makes the KG's event nodes *functionally* used by the agent (previously they
were primarily browsable in `kg_view.html`), matching this session's
established emphasis on "the knowledge graph is functional, not decorative."

## Method (documented, not tuned to any outcome)

Per hazard, a fixed set of raw indicators already used elsewhere in this
project (Layer 1's detection rules / `FEATURE_VARS`) is compared:
- `flash_flood`: cape, ivt, pwat, daily_precip_total, wind850_speed
- `heatwave`: tmax_c, heat_index_c, vpd_kpa, t2m_c, tmax_anomaly_c

Each feature is z-scored using the dataset's own mean/std (so CAPE in the
thousands doesn't silently dominate VPD in single digits), normalized
Euclidean distance is computed across shared features, and converted to a
0-100% score via `100/(1+distance)` — monotonically decreasing, 100% only at
distance 0. An event is excluded (with a stated reason, not silently
dropped) if fewer than half its features are available.

## Testing (14 new checks, matching the session's standing methodology)

### Core-math validation
Querying at an event's own exact coordinates and date must score **exactly
100%** — this was verified and is the foundational sanity check that the
z-score/distance formula is implemented correctly.

### A real finding, investigated before writing the test
The first exploratory run showed Jizan city center scoring only **0.8%**
similarity to the "08-23 extreme rain" event — on its *own* date. This looked
wrong until investigated: the KG event's coordinates (17.5N, 42.9E) are that
day's grid-cell **maximum** (the storm centroid), independently confirmed
~74km from Jizan's city-center coordinate (16.9N, 42.6E). Raw values at the
two points on the same day: `daily_precip_total` = 0.6mm at the city center
vs 254.9mm at the centroid. This is real, hyperlocal storm behavior, not a
bug — but it was surprising enough that it warranted a dedicated
`event_distance_from_city_km` field on every result and an explicit note, so
neither the agent nor the user mistakes a low same-day score for an error.

### A real positive match, found then independently verified
Sampling across all cities/hazards/dates found Mecca on 2025-08-04 scoring
46.1% similarity to the 07-25 Empty Quarter heat event, despite being 1,659km
apart and 8°C different in absolute tmax. Independently confirmed from the
raw source file: Mecca's tmax_c=45.95°C with anomaly +6.65°C is genuinely
comparable in *profile* (both anomalously hot for their own conditions) to
the Empty Quarter's tmax_c=53.75°C, anomaly +10.15°C — the z-scored
comparison correctly captures "both anomalous" rather than being fooled by
the raw temperature gap.

### Bypass and re-derivation
The Mecca/07-25 similarity score was independently hand-computed (raw values
+ the same cached mean/std) and matched the tool's own output exactly. In
`FULL_SYSTEM_AUDIT.py` Section I, it was re-derived a THIRD time — reading
both raw indicator vectors directly from the raw 5GB source files (bypassing
even the consolidated dataset) and recomputing mean/std fresh from the
consolidated dataset (not reusing `tools.py`'s cache) — and still matched.

## Live agent verification

Asked live: *"Is Mecca's weather on 2025-08-04 similar to any known extreme
heat event?"* — the agent correctly reported 46.1% similarity to the 07-25
event, explicitly noted the ~1,659km distance, and did not overstate the
match ("the similarity is moderate, not a direct match").

## Verification trail

- `agent/tools.py` — `similar_events_tool()`, `_get_vector()`, `_feature_stats()`.
- `agent/02_test_tools.py` — 14 new checks (73/73 total, up from 58).
- `FULL_SYSTEM_AUDIT.py` Section I — self-match and the Mecca finding both
  re-derived directly from raw source files, independent of `tools.py`.
- `agent/03_agent.py` — 4th tool wired into schema/prompt with an explicit
  instruction not to treat a low same-city/same-day score as an error.
