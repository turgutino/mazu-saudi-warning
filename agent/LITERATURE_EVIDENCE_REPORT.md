# MAZU Extension — literature_evidence_tool (7th agent tool)

## Motivation

The causal KG's citation coverage is genuinely limited: only 6-7 hand-verified
citations ground 5 mechanisms (`kg/causal/corpus.py`), each individually
extracted and verbatim-quote-checked (`kg/causal/02_extract_causal.py`). Some
city/hazard combinations have no formal mechanism link at all —
`region_risk_tool`'s own disclosed finding is that Jeddah is `at_risk_of`
heatwave but its `exposed_to` mechanisms have zero overlap with heatwave's
actual `driven_by` mechanisms. Until now, the only honest answer to "why does
Jeddah get heatwaves" was a flat "the KG doesn't have a specific link".

This extension was prompted by a real conversation with the project's
advisor about **Relink** (`Relink: Constructing Query-Driven Evidence Graph
On-the-Fly for GraphRAG`, GitHub: `DMiC-Lab-HFUT/Relink`), a technique
originally pitched (in a document the advisor shared) for building-intelligence
RAG scenarios. The advisor correctly pointed out that Relink itself is a
general, domain-agnostic technique — the building-intelligence framing was
just one applied write-up, not a limitation of the underlying idea. The
underlying idea (query-time, dynamically-assembled evidence from a loose
candidate pool, kept explicitly separate from a trusted "skeletal" graph
until human-reviewed) maps directly onto MAZU's own disclosed
citation-coverage gap.

**Relink's own reference implementation was read directly (not just its
paper abstract)** before deciding how to adapt it: `core/retriever.py` does
LLM-based topic-entity extraction, Neo4j beam search across both a formal KG
and a looser co-occurrence graph, and — critically —
`cover_co_occurrence_to_rel()`, where an **LLM predicts what relation a loose
co-occurrence represents**. That step is a generative, unverified inference,
which is exactly the category of risk this project has mechanically guarded
against everywhere else (`02_extract_causal.py`'s verbatim-quote gate; every
report's "never state a fact without a tool result" rule). `core/path_ranker.py`
also requires a separately-trained neural ranking model served over Redis.
Given MAZU's corpus size (12 literature entries, not the thousands of daily
logs Relink targets), that infrastructure is unjustified — so this tool
takes Relink's **idea**, not its stack: plain TF-IDF cosine similarity
(`sklearn`, already a project dependency) over the existing literature pool,
with every result mechanically labeled `verified: false`.

## What it does

`literature_evidence_tool(city, hazard)`:

1. Loads `kg/causal/corpus.py`'s full pool — 12 entries: the 7 original,
   formally-verified-and-KG-grounded citations, plus **5 new entries added
   for this tool** (see below), fits a `TfidfVectorizer` over their text.
2. Builds a query (`"{hazard} risk in {city}, Saudi Arabia"`, with the
   hazard's underscore normalized to a space — `flash_flood` → `flash flood`
   — otherwise it fails to match any corpus token at all; caught by testing,
   not assumed).
3. Cosine-similarity-ranks all 12 entries against the query, keeps the top 3
   above `LITERATURE_MIN_SIMILARITY = 0.08` (empirically chosen — see the
   score distribution below).
4. Returns each candidate with its real citation, URL, mechanism tag, the
   **exact, unmodified corpus text** (never an LLM paraphrase), a
   similarity score, an `already_in_formal_kg` flag, and `verified: false`
   on every single result — plus a top-level disclaimer.

This is strictly additive: it never writes to `kg_data.json` or
`causal_triples.json`, and the formal KG (60 nodes/183 edges) is completely
unchanged.

## The 5 new corpus entries — real, verified, freshly researched for this tool

Found via live web search, then **the actual abstract page was loaded and
read directly** (not trusted from a search-snippet summary) before writing
each paraphrase — the same discipline as the original 7 entries, extended:

| id | Citation | Mechanism tag | Why it was chosen |
|---|---|---|---|
| `ref_jeddah_uhi_addas2023` | Addas (2023), *Land* (MDPI) | `urban_heat_island` | Directly targets the disclosed Jeddah/heatwave KG gap |
| `ref_sw_orographic_alharbi2026` | Alharbi (2026), *Atmosphere* (MDPI) | `orographic_lifting` | SW Saudi Arabia flash-flood/monsoon mechanism, very recent (2026) |
| `ref_dust_transport_alzaid2024` | Alzaid, Anil & Aga (2024), *Atmosphere* (MDPI) | `cross_border_dust_transport` | Complements the existing Shamal citation with a source-region finding (Iraq/Syria) |
| `ref_sst_teleconnection_almaashi2024` | Almaashi, Hasanean & Labban (2024), *Atmosphere* (MDPI) | `sst_teleconnection` | Directly relevant to the dataset's own `sst_celsius` feature |
| `ref_jeddah2022_flood_sofia2024` | Sofia et al. (2024), *Water* (MDPI) | `orographic_lifting` | Real, documented Jeddah flood event (24 Nov 2022, "heaviest rainfall in the region's history") |

All 5 are MDPI open-access articles, chosen specifically because 4 other
promising candidates (Wiley, ScienceDirect, AMS, ResearchGate) returned
HTTP 402/403 when fetched directly — rather than write a paraphrase from an
unverified search-snippet summary for those, they were dropped and MDPI
open-access alternatives were found instead.

## A real bug found and fixed during testing

The first version constructed the query as `f"{hazard} risk in {city}..."`
using the raw hazard string (`"flash_flood"`, `"dust_storm"` — with
underscores). Since the corpus text uses natural language (`"flash flood"`,
`"dust storm"`), the underscored token matched **nothing**, and every
non-Jeddah, non-heatwave query silently collapsed to the same generic
top-5 result (driven only by common low-information words like "Saudi",
"Arabia", "risk"). Caught by manually inspecting a 6-combination score
printout before writing the formal tests, not assumed correct — fixed by
normalizing `hazard.replace("_", " ")`, then re-verified across all 24
city×hazard combinations. This is now a locked-in regression test.

## Real, disclosed finding: the "no candidate" path is correct but never exercised in practice

Across all 24 valid city×hazard combinations, the top-1 similarity score
ranges from **0.1421 to 0.2406** — always well above the 0.08 threshold.
Because the entire 12-entry corpus is Saudi-Arabia-weather-focused, every
valid query finds at least 2 candidates; the threshold's real job is
trimming weak 4th/5th-place tail matches (which do fall to ~0.06–0.08),
not gating a fully-empty result for these specific queries. The
below-threshold path is real code, not dead code — proven with a
deliberately off-topic synthetic query ("chocolate cake baking recipe"),
which scores exactly 0.0 and is correctly excluded. Reported honestly
rather than silently rounding up ("works for edge cases") — see
`FULL_SYSTEM_AUDIT.py` Section P and `agent/02_test_tools.py`'s dedicated
test for exactly this.

## Testing

18 new checks in `agent/02_test_tools.py` (169/169 total, up from 151):
field-shape checks, the `verified: false` invariant on every result, score
ordering, the underscore-normalization regression test, excerpt-integrity
checks (byte-for-byte match to raw `corpus.py` text), a bypass rebuilding
the TF-IDF matrix from scratch with a fresh vectorizer instance, the
off-topic-query dead-code check, error-shape parity with other tools,
determinism, and full 24-combination structural coverage.

`FULL_SYSTEM_AUDIT.py` Section P (new, 8 checks): independently reloads
`corpus.py` (bypassing every cache), rebuilds the TF-IDF matrix from
scratch, and confirms the live tool's top-3 candidates and scores match
exactly for a real query; confirms every returned excerpt is byte-identical
to the raw corpus text; independently re-derives the candidate count for
**all 24** city×hazard combinations (not a sample) and confirms exact
agreement with the live tool; and independently re-checks
`already_in_formal_kg` against a fresh read of `kg_data.json`'s
`grounded_by` edges.

## Live agent verification

Asked live: *"Why is Jeddah at risk of heatwaves? The knowledge graph
doesn't seem to have a mechanism linked to this city for heatwave — can you
check the literature for a candidate explanation?"* The agent correctly
called `causal_kg_tool`, `region_risk_tool`, and `literature_evidence_tool`
in sequence, explicitly confirmed the KG gap, surfaced the urban-heat-island
candidate with a hedged "literature suggests (unverified)" framing, and
never stated the candidate as an established fact — matching the system
prompt's strict requirement.

## Relationship to the two rejected point-estimate fixes and the deployed uncertainty tool

This is the third extension in the "how do we responsibly present something
short of full certainty" family, and follows the same pattern established by
`UNCERTAINTY_REPORT.md`: add a clearly-labeled, additive signal rather than
quietly upgrading an unverified thing into a verified one. Unlike the
isotonic-calibration and ensemble-averaging attempts (which changed a
decision-relevant number and were rejected), and like the uncertainty field
(which added context without touching `probability`), `literature_evidence_tool`
never writes into the trusted KG — every result is permanently `verified: false`
until a human reviews and manually promotes it through the same
`02_extract_causal.py` verbatim-quote pipeline the original 7 citations went
through.

## Performance impact

None on any existing tool, model, or KG value. This is a new, independent
7th tool; `causal_kg_tool`, `region_risk_tool`, and all other tools' outputs
are byte-for-byte unchanged.
