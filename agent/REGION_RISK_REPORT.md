# MAZU Extension — region_risk_tool: Closing the KG Utilization Gap

## Motivation

An honest self-audit of KG edge-type usage found that the agent's 4 tools
only actively queried 3 of the KG's 11 edge types (`driven_by`,
`contributes_to`, `grounded_by` — ~37 of 183 edges, ~20%). The
`at_risk_of`/`exposed_to` edges (37 more edges, region-level hazard
exposure) existed in the graph and were visible in `kg_view.html`, but no
tool ever read them — the opposite of the site's "functional, not
decorative" claim for that portion of the graph. This tool closes that gap.

## What it does

`region_risk_tool(city, date=None)` is **city-first** (the other 4 tools are
hazard-first — "why does flash_flood happen"). It answers "what should I
worry about in this city": which hazards a city is `at_risk_of`, which
mechanisms it is `exposed_to`, and — if a date is given — live forecast
probabilities for the hazards that have a trained model.

## Two real findings, both investigated and disclosed rather than hidden

### 1. Not every KG hazard has a forecast model
The KG has a 4th Hazard node, `coastal` ("Coastal / Marine Risk"), with a
`driven_by` mechanism (`moisture_transport`) but no trained forecast model
(only flash_flood/heatwave/dust_storm were trained). Several cities
(Jeddah, Jizan, Dammam) are `at_risk_of` it. The tool handles this
gracefully: `has_forecast_model: false`, and no `forecast` key is fabricated
for it — verified by test and by audit Section K.

### 2. A genuine, pre-existing KG data-quality gap (not introduced by this
### session's work)
Jeddah is `at_risk_of` heatwave, but Jeddah's `exposed_to` mechanisms
(`moisture_transport`, `ARST` — both flash-flood-relevant) have **zero
overlap** with heatwave's actual driving mechanisms (`subtropical_high`,
`thermal_low`). Traced to the original hand-encoded domain knowledge in
`kg/01_build_structural_kg.py`'s `REGION_HAZARD`/`REGION_MECH` dicts (written
before this session, not touched by today's dust_storm work) — a minor
inconsistency that predates all of today's changes. The tool's behavior is
to report an honest empty `mechanisms_affecting_this_city` list for that
combination rather than fabricating a mechanism or crashing; this is now
locked in by a dedicated test and independently re-confirmed in audit
Section K.

## Testing

23 new checks in `02_test_tools.py` (112/112 total, up from 90): a bypass
test against the raw KG JSON, the two findings above (both explicitly
asserted, not just tolerated), forecast-probability cross-checks against
independent direct `forecast_tool` calls (exact match, not "present"), a bad
date handled without silently dropping hazard info, and all 8 known cities
confirmed to resolve with at least 1 hazard.

`FULL_SYSTEM_AUDIT.py` Section K re-derives the Jizan hazard list, the
`coastal`-has-no-model handling, the Jeddah/heatwave gap, and a forecast
cross-check, all directly from the KG JSON / a fresh `forecast_tool` call —
independent of `region_risk_tool`'s own internals.

## Live agent verification

Asked live: *"What hazards should I worry about in Jizan, and what's the
current risk today (2025-08-23)?"* — the agent correctly called
`region_risk_tool`, then followed up with `causal_kg_tool`,
`conditions_tool`, and `similar_events_tool` to build a full answer; it
correctly reported "no trained forecast model" for the coastal hazard
instead of guessing, and correctly explained why the Aug 23 event scored low
similarity to itself (the hyperlocal-storm-centroid finding from
`similar_events_tool`'s own report). All 5 tools composed correctly together
in one live answer.

## KG utilization, before and after

| | Before this tool | After |
|---|---|---|
| Edge types actively queried by an agent tool | 3 of 11 | 5 of 11 |
| Edges actively queried | ~37 of 183 (~20%) | ~74 of 183 (~40%) |

Still not 100% (`correlates_with`, `sourced_from`, `manifests_as`,
`occurs_at`, `observed_value` remain unused by any tool) — disclosed here
rather than overstated, consistent with this project's standing practice of
reporting real numbers rather than rounding up.
