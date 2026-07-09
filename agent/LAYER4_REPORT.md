# MAZU Layer 4 — Explainable Warning Agent

## What this layer does

Wires the forecast model (Layer 2), the literature-grounded causal knowledge
graph (Layer 3), and a live-conditions readout into a single LLM agent
(DeepSeek, function calling) that answers natural-language risk questions
with grounded numbers and citations — never invented ones.

This is the layer that makes the knowledge graph *functional* rather than a
browsable exhibit: the agent programmatically queries it on every risk
question, the same way the competitor team's (戚语轩/Saudi-Graph) detection
engine queries theirs — closing that specific gap identified earlier.

## Architecture

```
User question (natural language)
        |
        v
DeepSeek (deepseek-chat, function calling, temperature=0)
        |
        +-- forecast_tool(city, target_date, hazard)
        |     -> loads the SAME saved, pre-verified sklearn models from
        |        Layer 2 (heatwave: neighbour-feature model, ROC-AUC 0.971;
        |        flash_flood: plain baseline, ROC-AUC 0.873)
        |
        +-- causal_kg_tool(hazard)
        |     -> queries kg/kg_data.json for driving mechanisms + Layer 3
        |        literature citations with verbatim evidence quotes
        |
        +-- conditions_tool(city, date)
        |     -> reads actual raw indicator values from the dataset
        |
        v
Final answer: probability + model ROC-AUC + mechanism + citation + recommendation
```

## Verification (deep, matching the session's established pattern)

### Step 1 — models retrained and cross-checked against Layer 2 reports
`01_train_and_save_models.py` retrains both production models using the
exact code paths already verified, then **refuses to save** if the new
ROC-AUC/PR-AUC drift from the previously reported numbers by more than
0.01. Result: heatwave 0.9706 (expected 0.971), flash_flood 0.8732
(expected 0.873) — both matched, saved.

### Step 2 — each tool tested independently before LLM wiring (27/27 passed)
`02_test_tools.py` — known events, negative controls, and error handling for
all three tools. Two real bugs were found and fixed during this step (not
hidden):

1. **causal_kg_tool bug**: the `driven_by` edge direction is
   `hazard -> mechanism` in the KG (verified against
   `kg/01_build_structural_kg.py`), but the query had source/target
   reversed, returning zero mechanisms for every hazard. Fixed and
   re-verified against the KG's own design (`flash_flood` now correctly
   returns `[ARST, moisture_transport, orographic_lift]`; `heatwave`
   returns `[subtropical_high, thermal_low]`).

2. **Tool interface flaw (UX/reasoning risk, not a code bug)**: the first
   design had `forecast_tool(city, date, hazard)` where `date` was "today"
   and the tool predicted `date+1`. Testing found this is an off-by-one
   trap for the LLM: a user asking "risk **on** 2025-08-23" would need the
   agent to correctly pass `date=2025-08-22`, an error-prone mental step.
   Redesigned to `forecast_tool(city, target_date, hazard)`, where
   `target_date` is the exact date being asked about and the tool
   internally looks up the prior day — removing this entire class of
   potential agent error at the interface level rather than hoping the LLM
   gets the arithmetic right every time.

3. **A third apparent anomaly was investigated and found NOT to be a bug**:
   an initial test assumed forecast probability should rank with raw
   next-day tmax across cities. It didn't (Spearman rho=0.38, not
   significant) — investigation showed our own `heatwave_day_flag` label
   (Layer 1) is defined by temperature **anomaly relative to local
   climatology**, not an absolute cutoff. Mecca's forecast probability
   (62.8%) correctly exceeded Riyadh's (3.1%) despite lower absolute
   temperature, because Mecca was +3.68°C above its own climatology that
   day while Riyadh was −2.42°C below its own. This is correct model
   behaviour consistent with our own label design, not an error — the test
   assumption was wrong, and was corrected rather than the tool.

### Step 3 — end-to-end agent tests (4/4 scenarios correct)

| Scenario | Result |
|---|---|
| Known flash-flood event (Jizan, 2025-08-23) | Correctly called forecast + KG tools; cited de Vries et al. (2013) and the Red Sea rainfall-contribution study with real quotes; honestly noted orographic lifting has no specific citation. |
| Known heatwave event (Mecca, 2025-07-25) | Called all 3 tools; probability (62.8%) and anomaly (+3.68°C) exactly matched independently-verified tool test output; cited 2 real sources correctly. |
| Out-of-range date (2026-03-15) | Did NOT hallucinate a probability; explained the 2025-only data range; proactively offered a valid alternative date. |
| Unknown city (Dubai) | Declined without even calling a tool (correctly used the enum-constrained city list); did not invent data for a city outside Saudi Arabia. |

## Honest limitations (disclosed)

- **Historical-data demo, not real-time.** All "forecasts" are made against
  the fixed 2025 dataset; "today" is whatever date the user specifies, not
  the actual current date. This is stated explicitly in the system prompt
  and the agent discloses it when relevant.
- **Only 8 named cities** are supported (those with defined coordinates);
  broader grid-point queries are not exposed to the agent in this layer.
- **orographic_lift remains uncited** (same disclosed gap as Layer 3) — the
  agent correctly states this mechanism without a fabricated citation
  rather than inventing one.
- **Flash-flood forecasts remain lower-confidence** (ROC-AUC 0.873, PR-AUC
  0.089 — see Layer 2) than heatwave forecasts (ROC-AUC 0.971) — the agent
  reports the model's own verified ROC-AUC with every answer so the user
  can judge reliability themselves, rather than presenting all forecasts
  with false uniform confidence.

## Files

```
agent/
  01_train_and_save_models.py   trains + verifies + persists production models
  saved_models/                 heatwave_model.joblib, flash_flood_model.joblib, model_meta.json
  tools.py                      the 3 agent tools
  02_test_tools.py              27 independent tool tests (all passing)
  03_agent.py                   DeepSeek function-calling orchestration
  LAYER4_REPORT.md              this file
```

## Security

The DeepSeek API key is read from `kg/causal/.deepseek_key` (gitignored,
never committed — same key file reused from Layer 3, verified absent from
every commit).
