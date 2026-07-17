# MAZU — Post-Layer-4 Extensions: Design Sources, Methodology, Results

Three extensions were added to the already-deployed Layer 4 agent after reviewing
external material shared for context: the MAESTRO multi-agent early-warning
paper (Wang et al., *npj Artificial Intelligence*, 2026, Zhejiang University —
a live, government-deployed system), and a recorded Kimi-AI consultation from
a related project about fine-tuning GraphCast for Saudi extreme weather. None
of that material was copied; the underlying *methodological ideas* were
extracted, re-derived against our own data, and independently tested. Each
extension below follows the session's
standing rule: build in isolation, write known-answer + negative-control +
error-case tests *before* trusting the change, investigate any test failure
as a possible real bug before assuming the test is wrong, only then wire into
the live agent, then verify with a real DeepSeek call.

All existing Layer 4 tests still pass. Total unit tests: **45/45** (up from
32/32 before this work — see `02_test_tools.py`), plus a new live ablation
experiment (`05_ablation_test.py`) with 4 real DeepSeek Q&A pairs run twice
each (8 live agent calls).

---

## Extension 1 — Terrain/elevation context on `forecast_tool`

**Source idea:** a Kimi-AI chat log (reviewed for context, not our own
material) discussing GraphCast's published limitation (§7.2.5 of the
GraphCast paper) that forecast error grows in high-elevation terrain,
because coarse global models under-resolve steep local relief. The
suggestion was to cross-check any forecast against a location's real
elevation and flag low-confidence terrain rather than silently trusting the
model everywhere.

**What we built:** `orography` (static surface elevation, from the same raw
5GB source used everywhere else in this project) was merged into
`data/mazu_dataset.nc` (`pipeline/07_add_orography.py`). `forecast_tool` now
returns `elevation_m` and, for cities at ≥1500 m, a `terrain_note` caveat.

**Real bug found and fixed during testing (not hypothetical — this is the
kind of thing "test deeply before trusting it" is for):** the first
implementation read elevation from the model's stride-2 (~20 km) feature
grid, matching whatever coarse cell the forecast model happens to use. Test
`Abha: elevation_m is a real, high value (mountain city)` FAILED: it returned
1333.7 m instead of Abha's known real elevation (~2270 m). Investigating
(not assuming the test was simply wrong) traced this to genuine steep local
relief in the Asir range — a full-resolution neighborhood dump showed
elevation swinging from 611 m to 2253 m within 3 grid cells around Abha, so
the *nearest coarse-grid cell* to Abha's coordinates happens to sit on a
foothill (1334 m), one full-resolution cell away from the true 2082 m peak.
This is not a coding bug; it is a real, previously undocumented
characteristic of the existing (already-deployed) forecast model: **in Abha
and similarly steep terrain, the model's own feature grid can substantially
under-represent local elevation** — precisely the situation this feature
exists to flag. Fix: elevation context is now read from the *full-resolution*
orography grid (a static geographic fact independent of which coarse cell
the model's features happen to sample), not the model's stride-2 grid.

**Test results (`02_test_tools.py`):** 5/5 new checks pass — Abha and Taif
(mountain cities) correctly flagged; Jeddah and Dammam (coastal, near sea
level) correctly NOT flagged; elevation values cross-checked against known
real-world figures for all 8 cities.

---

## Extension 2 — A/B ablation test: does the causal KG tool actually matter?

**Source idea:** an evaluation design comparing "with knowledge graph" vs
"without" to demonstrate the KG reduces hallucination. The MAESTRO paper
validates this exact methodology at publication scale (their Fig. 2c–d:
zero-shot / one-shot / two-shot / with-tool comparison, showing tool
grounding is what drives accuracy from 72% to 98%).

**What we built (`agent/05_ablation_test.py`):** the SAME live DeepSeek
agent, the SAME 4 "why does this hazard happen" questions, run twice each —
once with all 3 tools, once with `causal_kg_tool` removed from both the tool
list AND the system prompt (a parallel `NO_KG_SYSTEM_PROMPT` forbids
guessing, same as the main prompt's rule 1). Scoring is objective and
machine-checkable (keyword match against real KG mechanism/citation
vocabulary — not subjective LLM-judge scoring), specifically to keep this
reproducible rather than trusting a vibe.

**Results (`ABLATION_REPORT.md`, full transcripts saved):**

| Metric | WITH causal_kg_tool | WITHOUT causal_kg_tool |
|---|---|---|
| Answers naming a real driving mechanism | 4/4 | 0/4 |
| Answers citing real literature | 4/4 | 0/4 |
| Ungrounded mechanism claims (safety check) | 0/4 | 0/4 |

**Safety result: PASS.** With the tool removed, the agent did not fabricate
mechanism names from its own background LLM knowledge — in all 4 cases it
explicitly said it had no grounded causal explanation available (e.g. *"I
don't have a tool that provides physical mechanisms or literature
citations... I cannot responsibly answer that part of your question without
guessing"*). This is the critical result: it shows the system prompt's
strict grounding rule holds even when the relevant tool is entirely absent,
not just when the right tool happens to be available — i.e., the safety
property is robust to (not merely compatible with) the KG's presence.

---

## Extension 3 — Impact-based context (population reference)

**Source idea:** the MAESTRO paper's central framing — WMO's Impact-Based
Forecast and Warning Services (IBFWS) guidance — that warnings should be
interpretable in terms of human impact, not only hazard probability.

**What we built:** `agent/city_population.json`, sourced from the **Saudi
Census 2022 (GASTAT, General Authority for Statistics)** — verified via live
web search this session, not invented (Riyadh 9.06M, Jeddah 5.71M, Mecca
2.52M, Medina 1.85M, Dammam 1.57M, Taif 0.73M, Abha 422,243, Jizan 173,919;
Jizan has two public figures depending on boundary definition, the lower/more
conservative one is used and the discrepancy is disclosed in the file).
`forecast_tool` now returns `impact_context` with this population figure.

**Deliberately NOT done, to avoid overclaiming:** this is a *reference*
population, not an estimate of people who would be exposed to a specific
hazard — that would require exposure/vulnerability modeling (flood plains,
building density, etc.) this system does not perform. The field's own `note`
says so explicitly, and the agent's system prompt was updated with an
explicit rule: use this only to describe city scale (*"Riyadh, pop.
~9.06M"*), never state or imply a specific affected-person count. Verified
live: asking the agent "how many people live there" produced *"this is a
reference figure to help understand the scale of the city, not a modeled
exposure estimate"* — the disclaimer surfaced correctly in a real answer, not
just in the tool's returned JSON.

**Test results:** 8/8 checks — every one of the 8 cities resolves to a
population figure (no silent gaps), Riyadh and Jizan values checked against
their real, sourced figures, and the disclaimer text is checked (not just
presence of the field, but that it actually contains the "NOT a model
output... exposure" caveat).

---

## What was intentionally NOT done (scope discipline)

Two more ideas surfaced in the same review (CAP/ITU-T X.1303-style output
formatting; retrieval of similar historical events by embedding similarity)
were judged lower priority for a two-person student project and were not
built this round — flagged as optional future work, not silently dropped.

## Full verification trail

- `pipeline/07_add_orography.py` — merges static elevation into the dataset, self-verifying (asserts grid match before merge, re-opens and checks the result after).
- `agent/tools.py` — `forecast_tool` now returns `elevation_m`, `terrain_note`, `impact_context` in addition to the original fields; `causal_kg_tool` and `conditions_tool` unchanged.
- `agent/02_test_tools.py` — 45/45 passing (32 original + 13 new: 5 terrain + 8 impact-context).
- `agent/05_ablation_test.py` + `agent/ablation_results.json` + `agent/ABLATION_REPORT.md` — live 8-call experiment, full transcripts retained for audit.
- `agent/03_agent.py` — system prompt and `ask()` updated (the latter now accepts overridable prompt/schemas/funcs specifically so the ablation script reuses the real agent loop instead of a re-implementation that could drift from it).
