# =============================================================================
# MAZU — Layer 3: DeepSeek + Chain-of-Thought causal triple extraction
#
# Method (adapted from LLM4TyphoonKG, github.com/2BAIHAO/LLM4TyphoonKG):
# a LLM reads a literature passage and outputs structured causal triples.
#
# Anti-hallucination safeguard (this project's addition): every triple must
# include an EXACT VERBATIM QUOTE from the source text as evidence. After
# extraction, each quote is automatically checked against the source text —
# any triple whose quote cannot be found verbatim is REJECTED, not silently
# kept. This makes fabrication mechanically detectable, not just plausible.
#
# Output: kg/causal/causal_triples.json (verified triples only)
#         kg/causal/extraction_report.txt (full audit: accepted + rejected)
# =============================================================================

import os
import json
import re
import time
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(HERE, ".deepseek_key")
OUT_TRIPLES = os.path.join(HERE, "causal_triples.json")
OUT_REPORT = os.path.join(HERE, "extraction_report.txt")

import importlib.util
spec = importlib.util.spec_from_file_location("corpus", os.path.join(HERE, "corpus.py"))
corpus_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus_mod)
CORPUS = corpus_mod.CORPUS

SYSTEM_PROMPT = """You are a meteorological knowledge-graph extraction assistant.

You will be given ONE passage of scientific text about Saudi Arabia / Middle East
extreme weather mechanisms. Extract causal or mechanistic relationships that are
EXPLICITLY stated in the text. Do NOT use outside knowledge. Do NOT infer anything
beyond what the passage literally says.

For each relationship, output a JSON object with these exact fields:
  "subject": short noun phrase (the cause / driving factor)
  "relation": one of ["causes", "enhances", "precedes", "produces", "drives", "requires"]
  "object": short noun phrase (the effect / driven factor)
  "evidence_quote": an EXACT VERBATIM substring copied from the passage (word-for-word,
                     no paraphrasing) that supports this relationship. This quote will
                     be automatically checked against the source text — if it is not
                     an exact substring, the triple will be rejected.

Return ONLY a JSON array of such objects, nothing else. If the passage supports no
clear causal relationships, return an empty array []."""


def call_deepseek(client, text, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Passage:\n\n{text}"},
                ],
                temperature=0.0,
                max_tokens=1500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"    [retry {attempt+1}/{retries}] API error: {e}")
            time.sleep(2)
    raise RuntimeError("DeepSeek API failed after retries")


def parse_json_array(raw):
    """Extract a JSON array from the model output, tolerating markdown fences."""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.S)
    if m:
        raw = m.group(1)
    m2 = re.search(r"(\[.*\])", raw, re.S)
    if m2:
        raw = m2.group(1)
    return json.loads(raw)


def normalise(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_quote(quote, source_text):
    """Exact substring check, tolerant only to whitespace normalisation
    (not to paraphrasing) -- this is the anti-hallucination gate."""
    return normalise(quote) in normalise(source_text)


def main():
    with open(KEY_FILE) as f:
        api_key = f.read().strip()
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    all_accepted = []
    all_rejected = []
    report_lines = ["=" * 70, "MAZU Layer 3 — Causal triple extraction (DeepSeek + CoT)",
                    "=" * 70, f"Corpus entries: {len(CORPUS)}", ""]

    for entry in CORPUS:
        print(f"### {entry['id']} ({entry['citation']}) ###")
        raw = call_deepseek(client, entry["text"])
        try:
            triples = parse_json_array(raw)
        except Exception as e:
            print(f"  [ERROR] JSON parse failed: {e}\n  raw={raw[:300]}")
            report_lines.append(f"--- {entry['id']} : PARSE FAILED ---\n{raw[:500]}\n")
            continue

        report_lines.append(f"--- {entry['id']} ({entry['citation']}) ---")
        report_lines.append(f"Source: {entry['title']}")
        report_lines.append(f"URL: {entry['url']}")
        report_lines.append(f"Raw extracted: {len(triples)} candidate triples")

        for t in triples:
            q = t.get("evidence_quote", "")
            ok = verify_quote(q, entry["text"])
            record = {
                "subject": t.get("subject"), "relation": t.get("relation"),
                "object": t.get("object"), "evidence_quote": q,
                "source_id": entry["id"], "citation": entry["citation"],
                "url": entry["url"], "mechanism": entry["mechanism"],
            }
            if ok:
                all_accepted.append(record)
                report_lines.append(f"  [ACCEPTED] {t.get('subject')} --{t.get('relation')}--> {t.get('object')}")
                report_lines.append(f"             quote: \"{q}\"")
            else:
                all_rejected.append(record)
                report_lines.append(f"  [REJECTED — quote not verbatim in source] "
                                    f"{t.get('subject')} --{t.get('relation')}--> {t.get('object')}")
                report_lines.append(f"             claimed quote: \"{q}\"")
        report_lines.append("")

    with open(OUT_TRIPLES, "w", encoding="utf-8") as f:
        json.dump(all_accepted, f, ensure_ascii=False, indent=2)

    report_lines.append("=" * 70)
    report_lines.append(f"TOTAL: {len(all_accepted)} accepted, {len(all_rejected)} rejected "
                        f"({len(all_accepted)+len(all_rejected)} candidates)")
    if len(all_accepted) + len(all_rejected) > 0:
        rate = 100 * len(all_accepted) / (len(all_accepted) + len(all_rejected))
        report_lines.append(f"Verbatim-quote verification pass rate: {rate:.1f}%")
    report = "\n".join(report_lines)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\n[SAVED] {OUT_TRIPLES}")
    print(f"[SAVED] {OUT_REPORT}")


if __name__ == "__main__":
    main()
