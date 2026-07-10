# MAZU Extension — cap_alert_tool: CAP 1.2 Alert Generation

## Motivation

Every prior tool speaks to a human (natural-language answers) or to the LLM
agent itself (structured JSON). None of them speak to real warning
infrastructure. **CAP (Common Alerting Protocol) 1.2** is the OASIS
international standard for machine-readable public warnings — the wire
format national alerting systems (US IPAWS, EU alert networks) use, and the
format MAZU's own national early-warning framework is built on. Without CAP
support, the system could demonstrate scientific correctness but not
operational integration. This tool closes that gap: it converts a
`forecast_tool` probability into a real, standards-compliant CAP 1.2 XML
alert, ready to plug into siren/SMS/broadcast infrastructure.

## What it does

`cap_alert_tool(city, target_date, hazard)` calls `forecast_tool`
internally (not a reimplementation — the exact same probability, elevation,
and reflexive-check output), then maps that result onto CAP 1.2's required
fields and serializes valid, namespaced XML via `xml.etree.ElementTree`
(chosen over string formatting specifically so free-text fields are
correctly escaped, not hand-quoted).

- **`severity`** (Minor/Moderate/Severe/Extreme) is derived from
  `DetectionEngine`'s own percentile-grounded severity thresholds (the same
  ones `_reflexive_check` cross-checks against) — not a second, independently
  invented cutoff scale.
- **`certainty`** (Likely/Possible) is derived from the reflexive check's own
  `consistency` field: "Likely" only when the ML model AND the independent
  rule-based detection engine both agree the day is elevated-risk
  (`consistent_elevated`); any disagreement between the two signals
  downgrades to "Possible" — CAP's own semantics for "Likely" (>50%
  confidence) shouldn't be claimed off one unconfirmed signal.
- **`event`** is read from the KG's own Hazard node `label`
  (e.g. "Flash Flood / Wadi Flooding"), not a hardcoded string, so it stays
  in sync with the KG.
- Below the hazard's own lowest severity threshold, **no alert is issued at
  all** (`alert_warranted: false` with a `reason`) — matching how real
  warning systems behave (not every day gets a CAP message), rather than
  emitting a spuriously "Minor" alert for a 2% probability day.

## Honesty constraint: `status` is always `"Exercise"`, never `"Actual"`

This system runs entirely on the 2025 historical dataset, not a live feed
(see `tools.py`'s own top-of-file scope note). CAP's `status` field exists
precisely to distinguish real operational alerts from drills — hardcoding
`Actual` here would misrepresent the system to anything that ingested this
XML. Verified by a dedicated test and independently by audit Section L4,
which parses the real generated XML (not the Python dict) across all 3
hazards.

## A real finding surfaced during testing (investigated, not hidden)

While building test cases, `cap_alert_tool("Jizan", "2025-08-20",
"flash_flood")` returned `alert_warranted: false` (probability 36.9%,
flash_flood's CAP threshold is 50%) — yet the SAME call's
`reflexive_check.consistency` is `"consistent_elevated"` (both the model and
the independent rule-based engine agree the day is elevated, since that
check uses a lower, separate threshold of 0.30). This is not a bug: the
reflexive check answers "is this worth a caution note in the explanation"
(a lower bar), while CAP alert issuance answers "is this worth a formal
public warning" (a deliberately higher bar) — the two thresholds serve
different purposes and are not meant to coincide. Confirmed live: asked the
agent this exact question, it correctly explained both facts together
(elevated per the cross-check, but below the alert threshold) rather than
treating the mismatch as a contradiction.

## Testing

22 new checks in `02_test_tools.py` (134/134 total, up from 112):
XML-parses-and-round-trips checks (not string-matching), an XML-element-level
check that `<status>` is always `Exercise`, a probability cross-check against
an independent direct `forecast_tool` call, both certainty branches
(`Likely` from a genuine `consistent_elevated` day, `Possible` from a genuine
`model_higher_than_detection` disagreement — both real, found dates, not
constructed), the no-alert-below-threshold path, and error propagation
(unknown city / out-of-range date produce the *same* error dict as calling
`forecast_tool` directly, not a reworded copy).

`FULL_SYSTEM_AUDIT.py` Section L (6 checks, 90/90 total) independently
re-derives the severity mapping straight from `DetectionEngine.RULES`
(bypassing `tools.py`'s own `_cap_severity` helper), re-parses the actual XML
across all 3 hazards to confirm `<status>` and `<event>`, and cross-checks
`<event>` against the KG's hazard node label read directly from
`kg_data.json` — independent of `cap_alert_tool`'s own internals throughout.

## Live agent verification

Asked live: *"Generate a formal CAP alert for the flash flood risk in Jizan
on 2025-03-27."* — the agent called `forecast_tool`, `causal_kg_tool`, then
`cap_alert_tool`, and composed a correct answer: the full CAP XML, the
94.16% probability, the `model_higher_than_detection` reflexive-check
caveat, and an explicit note that this is an `Exercise`-status demo alert.

Asked a 2nd time for a known sub-threshold day (2025-08-20): the agent
correctly reported "no CAP alert issued" with the exact threshold
(36.92% < 50%), and did **not** fabricate an XML message — confirmed from
the raw tool-call trace, not just the prose answer.

## Performance impact

None on the model/KG/detection layers — `cap_alert_tool` is a pure output-
layer transformation on top of `forecast_tool`'s existing return value; no
model, feature, or KG logic was touched. All 134 unit tests and all 90 audit
checks (covering every prior extension, not just this one) pass unchanged.
