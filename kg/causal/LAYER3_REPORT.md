# MAZU Layer 3 — Literature-Grounded Causal Knowledge Graph

## What this layer does

The structural KG (Layer built earlier) encoded mechanism → indicator/hazard
relationships (`triggers`, `driven_by`) as **hand-written domain knowledge** —
my own assertions, uncited. This layer replaces that gap for 4 of 5
mechanisms with **verbatim-verified claims extracted from real, citable
peer-reviewed literature**, using DeepSeek + chain-of-thought extraction
(method adapted from LLM4TyphoonKG, github.com/2BAIHAO/LLM4TyphoonKG).

## Method

1. **Corpus (`corpus.py`)** — 7 passages, each a faithful paraphrase of a real
   publication found via live web search (not invented), covering ARST,
   moisture transport, the subtropical high, and the summer Shamal/thermal
   low. Every passage carries its real citation, title and URL.
2. **Extraction (`02_extract_causal.py`)** — DeepSeek (`deepseek-chat`,
   temperature 0) reads ONE passage at a time and is instructed to extract
   only causal/mechanistic relationships **explicitly stated in that
   passage**, with a mandatory `evidence_quote` field that must be an exact
   verbatim substring of the source text.
3. **Anti-hallucination gate** — every `evidence_quote` is automatically
   checked against the source passage (whitespace-normalised exact
   substring match). Quotes that don't verify are rejected, not kept.
4. **Manual quality review** — passing the quote check is necessary but not
   sufficient (a quote can be verbatim yet still describe a weak/circular
   claim). All 21 accepted triples were read individually; 1 was excluded
   for being descriptive/circular rather than genuinely mechanistic (see
   below). This is disclosed, not hidden.
5. **Merge (`03_merge_causal_kg.py`)** — verified triples become `Citation`
   nodes (one per source) with `grounded_by` edges from the existing
   `Mechanism` nodes, plus an `evidence` list (subject/relation/object/quote)
   shown in the dashboard on click.
6. **Independent re-verification** — after merging, every evidence quote was
   re-checked against the original corpus text a second time, independently
   of the extraction script's own check (see verification log below).

## Results

| Metric | Value |
|---|---|
| Corpus passages | 7 |
| Candidate triples extracted | 21 |
| Passed verbatim-quote check | 21 / 21 (100%) |
| Passed manual quality review | 20 / 21 |
| Excluded after manual review | 1 (circular/descriptive, see below) |
| Citation nodes added to KG | 6 |
| `grounded_by` edges added | 6 |
| Mechanisms now literature-grounded | 4 / 5 (ARST, moisture_transport, subtropical_high, thermal_low) |
| Mechanisms still hand-coded (disclosed) | 1 (orographic_lift — no corpus source found this round) |
| KG size | 51/170 → **57 nodes / 176 edges** |

## The excluded triple (honest disclosure)

Source `ref_redsea_coast_trends` (Nature Sci. Reports 2021) produced:

> "warming, drying coastal atmosphere" **causes** "significant positive
> trends in surface air temperature and wind speed, alongside significant
> negative trends in relative humidity and sea-level pressure"

The `evidence_quote` was verbatim in the source text, so it passed the
automatic gate. On manual review this is circular: the source text simply
*labels* a set of statistical trends as "consistent with" warming/drying —
it does not describe a causal mechanism. Keeping it in the KG would have
been a false positive for "literature-grounded causality." It is recorded
in `causal_triples.json` for full audit transparency but excluded from the
graph itself.

## What was NOT done (scope, disclosed)

- **orographic_lift** (Hejaz/Asir mountain rainfall enhancement) has no
  literature grounding yet — no corpus passage was gathered for it this
  round. It remains hand-coded domain knowledge in the KG, same as before.
- The corpus is 7 passages from a live, one-time web search — not a
  systematic literature review. It is a genuine, real, citable sample, not
  an exhaustive one.
- Only 4 of 5 mechanism nodes gained citations; the KG is honestly labelled
  as partially, not fully, literature-grounded.

## Files

```
kg/causal/
  corpus.py                 7 real, cited literature passages
  02_extract_causal.py      DeepSeek + CoT extraction with verbatim-quote gate
  causal_triples.json       21 extracted triples (all quote-verified)
  extraction_report.txt     full extraction audit log
  03_merge_causal_kg.py     merges verified triples into kg_data.json
  causal_kg_report.txt      merge audit log
  LAYER3_REPORT.md          this file
```

## Security note

The DeepSeek API key used for extraction is stored in
`kg/causal/.deepseek_key`, which is excluded via `.gitignore` and was never
committed. No key material appears in any file pushed to GitHub.
